"""Phase 3: unseen attack from unseen hosts, matched-FPR recall, feasibility and leakage."""

import numpy as np
import pandas as pd
import pytest

from cyberdefense.baselines.experiment import run_all as run_baselines
from cyberdefense.config import BaselineConfig, ExperimentConfig, GNNConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.protocols import (
    attack_host_feasibility,
    attack_host_holdout,
    attack_host_holdouts,
    holdout_hosts,
    holdout_label,
    holdout_mask,
    target_mask,
)
from cyberdefense.data.schema import SPLITS, load_split
from cyberdefense.gnn.audit import audit_graph
from cyberdefense.gnn.experiment import graph_for
from cyberdefense.gnn.experiment import run_all as run_gnns
from cyberdefense.metrics import matched_fpr_metrics, wilson_interval
from cyberdefense.reporting import headline_table, report_tables

MITM_HOSTS = ["172.16.0.1", "172.16.0.2"]


@pytest.fixture
def splits(tmp_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Synthetic splits where mitm comes only from two dedicated hosts (as in ToN-IoT)."""
    cfg = tmp_cfg.model_copy(
        update={"dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    out = {}
    for name in SPLITS:
        df = load_split(cfg.dataset, name).copy()
        mitm = (df["attack_type"] == "mitm").to_numpy()
        df.loc[mitm, "src_ip"] = np.resize(MITM_HOSTS, int(mitm.sum()))
        out[name] = df
    return out


def relaxed(cfg: ExperimentConfig, **baseline) -> ExperimentConfig:
    b = {
        # Synthetic ddos/scanning come from every host, so holding them out with their
        # senders leaves only the 2 mitm hosts: these thresholds keep mitm only.
        "attack_host_min_test_flows": 1,
        "attack_host_min_train_attack_types": 2,
        "attack_host_min_train_attacker_hosts": 5,
        "protocols": ["unseen_attack_host"],
        "tasks": ["binary"],
        "models": ["random_forest"],
        "random_forest": {"n_estimators": 10},
    }
    return cfg.model_copy(update={"baseline": BaselineConfig(**(b | baseline))})


def test_holdout_carries_every_sender(splits: dict) -> None:
    holdout = attack_host_holdout(splits, "mitm")
    assert holdout == "mitm@" + ",".join(MITM_HOSTS)
    assert holdout_hosts("unseen_attack_host", holdout) == MITM_HOSTS
    assert holdout_label(holdout) == "mitm (2 hosts)"
    for df in splits.values():
        removed = holdout_mask(df, "unseen_attack_host", holdout)
        touches = df["src_ip"].isin(MITM_HOSTS) | df["dst_ip"].isin(MITM_HOSTS)
        assert (removed == (touches | (df["attack_type"] == "mitm"))).all()


def test_target_is_the_type_plus_held_hosts_normal_traffic() -> None:
    df = pd.DataFrame(
        {
            "src_ip": ["h", "h", "h", "x", "x"],
            "dst_ip": ["v", "v", "v", "h", "v"],
            "attack_type": ["mitm", "ddos", "normal", "normal", "ddos"],
            "label": [1, 1, 0, 0, 1],
        }
    )
    target = target_mask(df, "unseen_attack_host", "mitm@h")
    # mitm yes; h's ddos (a seen type) no; h's own normal traffic yes (for its FPR).
    assert target.tolist() == [True, False, True, True, False]
    assert holdout_mask(df, "unseen_attack_host", "mitm@h").tolist() == [1, 1, 1, 1, 0]
    with pytest.raises(ValueError, match="<type>@<hosts>"):
        holdout_mask(df, "unseen_attack_host", "mitm")


def test_feasibility_skips_instead_of_forcing(splits: dict) -> None:
    f = attack_host_feasibility(splits, "mitm")
    assert f["held_hosts"] == 2
    assert f["train_attack_types"] == 2  # ddos and scanning remain
    assert f["test_target_flows"] > 0
    assert attack_host_holdouts(splits, ["mitm"], min_test_flows=1) == []  # needs 3 types
    assert attack_host_holdouts(
        splits, ["mitm"], min_test_flows=1, min_train_attack_types=2, min_train_attacker_hosts=1
    ) == [f["holdout"]]
    assert attack_host_holdouts(splits, ["mitm"], min_test_flows=10**6) == []


def test_matched_fpr_reads_every_model_at_the_same_test_fpr() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(200)].astype(int)
    held = np.r_[np.zeros(1100), np.ones(100)].astype(bool)
    for scores in (rng.random(1200) + y, rng.random(1200) * 0.1 + y * 0.05):
        out = matched_fpr_metrics(y_test=y, test_scores=scores, fprs=[0.01], test_holdout=held)
        assert out["matched_fpr_at_fpr1"] <= 0.01
        assert out["holdout_tpr_lo_at_fpr1"] <= out["holdout_tpr_at_fpr1"]
        assert out["holdout_tpr_at_fpr1"] <= out["holdout_tpr_hi_at_fpr1"]
    lo, hi = wilson_interval(156, 156)
    assert 0.97 < lo < 1.0 == hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_held_out_senders_are_isolated_and_audited(tmp_cfg: ExperimentConfig, splits) -> None:
    holdout = attack_host_holdout(splits, "mitm")
    data = graph_for(tmp_cfg, splits, "unseen_attack_host", holdout)  # validates
    result = audit_graph(data, splits, tmp_cfg)
    failed = [k for k, v in result["checks"].items() if not v["passed"]]
    assert not failed
    assert result["checks"]["held_out_host_has_no_inputs"]["passed"]
    assert result["checks"]["validate_graph:held_out_host_has_no_training_edges"]["passed"]
    assert result["exposure"]["target_src_is_known_attacker"] == 0.0


def test_baselines_and_gnns_share_the_protocol(tmp_cfg: ExperimentConfig, splits) -> None:
    cfg = relaxed(tmp_cfg)
    base = run_baselines(cfg, splits)
    holdouts = set(base["holdout"])
    assert holdouts == {attack_host_holdout(splits, "mitm")}  # ddos/scanning infeasible
    for col in ("test_holdout_tpr_at_fpr1", "test_holdout_tpr_lo_at_fpr0p25", "test_tpr_at_fpr1"):
        assert col in base.columns

    gnn_cfg = cfg.model_copy(
        update={
            "gnn": GNNConfig(
                models=["mlp", "sage"],
                protocols=["unseen_attack_host"],
                holdouts=["mitm"],
                seeds=[42],
                hidden_dim=16,
                gat_heads=2,
                max_epochs=2,
                patience=2,
                batch_size=64,
            )
        }
    )
    gnn = run_gnns(gnn_cfg, splits)
    assert set(gnn["holdout"]) == {attack_host_holdout(splits, "mitm")}
    mixed = pd.concat([base, gnn], ignore_index=True)
    table = headline_table(mixed, [])
    assert "unseen attack+host mean recall@matched 1% test FPR" in table.columns
    assert {"random_forest", "mlp", "sage"} <= set(table.index)
    tables = report_tables(mixed)
    title = (
        "unseen_attack_host: recall on the held-out type from unseen hosts"
        " at matched 1% actual test FPR"
    )
    assert "mitm (2 hosts)" in tables[title].index
