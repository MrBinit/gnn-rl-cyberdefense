"""Edge-classification GNNs: models, leakage guards, training, evaluation and reporting."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV

from cyberdefense.config import ExperimentConfig, GNNConfig, GraphConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.protocols import attacker_hosts
from cyberdefense.data.ton_iot import SPLITS, load_split
from cyberdefense.gnn.experiment import experiment_name, graph_for, run_all
from cyberdefense.gnn.models import EdgeClassifier, node_inputs, structure
from cyberdefense.gnn.train import class_weights, predict_proba, prepare_inputs
from cyberdefense.metrics import f1_threshold
from cyberdefense.reporting import headline_table, label_runs, load_summaries

SMALL = {
    "hidden_dim": 16,
    "gat_heads": 2,
    "max_epochs": 3,
    "patience": 2,
    "batch_size": 64,
    "seeds": [42],
}


@pytest.fixture
def splits(tmp_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    cfg = tmp_cfg.model_copy(
        update={"ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    return {name: load_split(cfg.ton_iot, name) for name in SPLITS}


def small_cfg(cfg: ExperimentConfig, **gnn) -> ExperimentConfig:
    return cfg.model_copy(update={"gnn": GNNConfig(**(SMALL | gnn))})


def test_class_weights_are_balanced() -> None:
    w = class_weights(torch.tensor([0, 1, 1, 1]), 3)
    assert torch.allclose(w, torch.tensor([2.0, 2 / 3, 0.0]))  # n / (k * n_c), k = 2 present


def test_node_inputs_and_structure() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 2]])
    attr = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    x = node_inputs(torch.ones(4, 1), edge_index, attr, 4)
    # [x, mean out, mean in, has_out, has_in]; node 3 has no edges -> only x.
    assert x[0].tolist() == [1.0, 2.0, 5.0, 1.0, 1.0]
    assert x[3].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    s = structure(edge_index, 4)
    assert s.shape[1] == 2  # 0-1 both ways, duplicates and the self-loop removed
    assert not (s[0] == s[1]).any()


def test_message_passing_ignores_val_and_test_edges(
    tmp_cfg: ExperimentConfig, splits: dict
) -> None:
    """train_only: changing val/test flows must not change any node embedding."""
    data = graph_for(tmp_cfg, splits, "random", None)
    before = prepare_inputs(data)
    other = data.clone()
    other.edge_attr[~data.mp_mask] += 100.0
    other.edge_index[:, ~data.mp_mask] = other.edge_index[:, ~data.mp_mask].flip(0)
    after = prepare_inputs(other)
    assert torch.equal(before.node_in, after.node_in)
    assert torch.equal(before.structure, after.structure)

    torch.manual_seed(0)
    model = EdgeClassifier(
        "sage", before.node_in.shape[1], before.edge_attr.shape[1], 2, GNNConfig()
    )
    train_idx = data.train_mask.nonzero().squeeze(1)
    assert torch.equal(
        predict_proba(model, before, train_idx), predict_proba(model, after, train_idx)
    )


def test_held_out_host_has_no_node_information(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    host = attacker_hosts(splits, 1)[0]
    data = graph_for(tmp_cfg, splits, "unseen_host", host)
    inputs = prepare_inputs(data)
    i = data.node_ips.index(host)
    assert inputs.node_in[i].tolist() == [1.0] + [0.0] * (inputs.node_in.shape[1] - 1)
    assert not ((inputs.structure[0] == i) | (inputs.structure[1] == i)).any()


def test_f1_threshold_matches_baseline_tuner() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = (x[:, 0] + rng.normal(size=2000) > 0.5).astype(int)
    model = LogisticRegression().fit(x[:1000], y[:1000])
    expected = TunedThresholdClassifierCV(FrozenEstimator(model), scoring="f1").fit(
        x[1000:], y[1000:]
    )
    assert f1_threshold(y[1000:], model.predict_proba(x[1000:])[:, 1]) == pytest.approx(
        expected.best_threshold_
    )


def test_run_all_every_model_and_protocol(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    host = attacker_hosts(splits, 1)[0]
    cfg = small_cfg(tmp_cfg, holdouts=["mitm", host])
    cfg = cfg.model_copy(
        update={"baseline": cfg.baseline.model_copy(update={"unseen_host_min_attack_flows": 1})}
    )
    summary = run_all(cfg, splits)

    # random x {binary, multiclass} + mitm + host, for 4 models, 1 seed.
    assert len(summary) == 4 * 4
    assert set(summary["model"]) == {"mlp", "gcn", "sage", "gat"}
    out = cfg.paths.results_dir / "gnn"
    assert (out / "summary.csv").exists()
    assert (out / "summary_aggregated.csv").exists()
    for col in ("test_recall_at_fpr0p25", "test_holdout_recall_at_fpr1", "test_holdout_roc_auc"):
        assert col in summary.columns
    run = json.loads(
        (
            out
            / "unseen_host/binary"
            / f"unseen_host__binary__gat__class_weight__seed42__holdout_{host}.json"
        ).read_text()
    )
    assert run["graph"]["message_passing"] == "train_only"
    assert 1 <= run["best_epoch"] <= run["epochs_run"] <= 3
    assert len(run["history"]) == run["epochs_run"]
    assert run["holdout_test_attack_flows"] > 0

    # Resume reuses the finished runs instead of retraining.
    again = run_all(
        cfg.model_copy(update={"gnn": cfg.gnn.model_copy(update={"resume": True})}), splits
    )
    pd.testing.assert_frame_equal(
        again.sort_values("fit_seconds").reset_index(drop=True),
        summary.sort_values("fit_seconds").reset_index(drop=True),
        check_like=True,
    )

    # Mixed with a "baseline" summary, the headline table has one row per model.
    (cfg.paths.results_dir / "baselines").mkdir()
    summary.assign(model="random_forest").to_csv(
        cfg.paths.results_dir / "baselines" / "summary.csv", index=False
    )
    mixed = load_summaries(
        [cfg.paths.results_dir / "baselines" / "summary.csv", out / "summary.csv"]
    )
    table = headline_table(mixed, ["xss"])
    assert set(table.index) == {"random_forest", "mlp", "gcn", "sage", "gat"}
    assert "unseen attack mean recall@0.25%FPR" in table.columns
    assert "multiclass macro-F1" in table.columns


def test_experiment_name_and_labels(tmp_cfg: ExperimentConfig) -> None:
    assert experiment_name(tmp_cfg) == "gnn"
    cfg = tmp_cfg.model_copy(
        update={"graph": GraphConfig(message_passing="transductive", node_features="degree")}
    )
    assert experiment_name(cfg) == "gnn_transductive_degree"
    df = pd.DataFrame({"model": ["xgboost"]})
    assert label_runs(df, "baselines")["model"].iloc[0] == "xgboost"
    assert label_runs(df, "baselines_tuned_disjoint")["model"].iloc[0] == "xgboost [tuned_disjoint]"
    with pytest.raises(ValueError, match="gat_heads"):
        GNNConfig(hidden_dim=10, gat_heads=4)


def test_headline_excludes_tuning_holdouts_for_every_model() -> None:
    rows = [
        {
            "protocol": "unseen_attack",
            "task": "binary",
            "model": m,
            "imbalance": "cw",
            "holdout": h,
            "tuned_for": False,
            "seed": s,
            "test_holdout_recall_at_fpr1": r,
            "test_holdout_roc_auc": 0.9,
        }
        for m in ("a", "b")
        for s in (1, 2)
        for h, r in (("xss", 0.0), ("mitm", 0.8), ("scanning", 0.6))
    ]
    table = headline_table(pd.DataFrame(rows), ["xss"])
    assert table.loc["a", "unseen attack mean recall@1%FPR"].startswith("0.7000")
    assert table.loc["b", "unseen attack worst recall@1%FPR"].startswith("0.6000")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU (MPS)")
def test_run_all_on_mps(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    cfg = small_cfg(tmp_cfg, device="mps", protocols=["random"], tasks=["binary"])
    summary = run_all(cfg, splits)
    assert len(summary) == 4
    assert summary["test_f1"].between(0, 1).all()
