"""Phase 2: unseen-pair protocol, host prior, ablations, self-loop FPRs and the audit."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from cyberdefense.baselines.experiment import iter_specs as baseline_specs
from cyberdefense.config import ExperimentConfig, GNNConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.protocols import attacker_hosts, holdout_mask, pair_fold, pair_holdouts
from cyberdefense.data.schema import SPLITS, load_split
from cyberdefense.gnn.audit import audit_graph, training_ignores_test_labels
from cyberdefense.gnn.experiment import (
    ablation_models,
    experiment_name,
    graph_for,
    run_all,
    self_loop_fprs,
)
from cyberdefense.gnn.models import EdgeClassifier, rewire, structure
from cyberdefense.gnn.shortcut import host_prior_proba
from cyberdefense.gnn.train import without_self_loops
from cyberdefense.reporting import headline_table

SMALL = {"hidden_dim": 16, "gat_heads": 2, "max_epochs": 2, "patience": 2, "batch_size": 64}


@pytest.fixture
def splits(tmp_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    cfg = tmp_cfg.model_copy(
        update={"dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    return {name: load_split(cfg.dataset, name) for name in SPLITS}


def with_gnn(cfg: ExperimentConfig, **gnn) -> ExperimentConfig:
    return cfg.model_copy(update={"gnn": GNNConfig(**(SMALL | {"seeds": [42]} | gnn))})


def test_pair_folds_keep_both_directions_together() -> None:
    df = pd.DataFrame({"src_ip": ["a", "b", "a", "c"], "dst_ip": ["b", "a", "c", "a"]})
    folds = pair_fold(df, 5)
    assert folds[0] == folds[1]
    assert folds[2] == folds[3]
    masks = [holdout_mask(df, "unseen_pair", h) for h in pair_holdouts(5)]
    assert (np.sum(masks, axis=0) == 1).all()  # every row in exactly one fold
    with pytest.raises(ValueError, match="pair_fold"):
        holdout_mask(df, "unseen_pair", "fold1")


def test_baselines_and_gnns_share_pair_holdouts(tmp_cfg: ExperimentConfig) -> None:
    cfg = tmp_cfg.model_copy(
        update={
            "baseline": tmp_cfg.baseline.model_copy(
                update={"protocols": ["unseen_pair"], "tasks": ["binary"]}
            )
        }
    )
    specs = baseline_specs(cfg, [], [])
    assert {s.holdout for s in specs} == set(pair_holdouts(5))


def test_host_prior_uses_training_labels_only() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 1, 3]])
    y = torch.tensor([1, 1, 0, 0, 0])
    train = torch.tensor([True, True, True, False, False])
    p = host_prior_proba(edge_index, y, train, 4, 2)[:, 1]
    assert p[0] == pytest.approx(1.0 / 1.0 * 0.5 + 1.0 * 0.5)  # 0 sent 2/2 attacks; 1 got 2/2
    assert p[3] == pytest.approx(1.0)  # node 2 sent nothing in train -> only dst 1's history
    assert p[4] == pytest.approx(1.0)  # dst 3 unseen -> src 0's history only
    y_test_flipped = y.clone()
    y_test_flipped[~train] = 1 - y_test_flipped[~train]
    assert torch.equal(p, host_prior_proba(edge_index, y_test_flipped, train, 4, 2)[:, 1])


def test_rewire_keeps_degrees_and_changes_neighbours() -> None:
    torch.manual_seed(0)
    ei = torch.randint(0, 40, (2, 200))
    s = structure(ei, 40)
    r = rewire(s, 40, seed=1)
    assert torch.equal(r, rewire(s, 40, seed=1))
    deg = lambda e: torch.bincount(e[0], minlength=40)  # noqa: E731
    assert (deg(s) - deg(r)).abs().sum() <= 0.2 * deg(s).sum()  # roughly kept
    assert not torch.equal(s, r)


def test_without_self_loops_only_shrinks_training_masks() -> None:
    data = Data(
        x=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 1, 2, 2]]),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
        mp_mask=torch.tensor([True, True, False, False]),
        num_nodes=3,
    )
    out = without_self_loops(data)
    assert out.train_mask.tolist() == [True, False, False, False]
    assert out.mp_mask.tolist() == [True, False, False, False]
    assert torch.equal(out.test_mask, data.test_mask)  # test (incl. its self-loop) unchanged


def test_self_loop_fprs() -> None:
    y = np.array([0, 0, 0, 1])
    loop = np.array([True, True, False, False])
    out = self_loop_fprs(
        y, np.array([0.9, 0.1, 0.2, 0.8]), (y * 0) + [1, 0, 0, 1], loop, {"k": 0.5}
    )
    assert out == {
        "self_loop_fpr": 0.5,
        "self_loop_fpr_at_k": 0.5,
        "non_self_loop_fpr": 0.0,
        "non_self_loop_fpr_at_k": 0.0,
    }


def test_ablation_models_and_names(tmp_cfg: ExperimentConfig) -> None:
    all_models = ["host_prior", "mlp", "gcn", "sage", "gat"]
    assert ablation_models(with_gnn(tmp_cfg, models=all_models)) == all_models
    cfg = with_gnn(tmp_cfg, models=all_models, ablation="no_propagation")
    assert ablation_models(cfg) == ["gcn"]
    assert experiment_name(cfg) == "gnn_no_propagation"
    cfg = with_gnn(tmp_cfg, models=all_models, ablation="topology_only")
    assert ablation_models(cfg) == ["gcn", "sage", "gat"]
    model = EdgeClassifier("gcn", 5, 7, 2, GNNConfig(**SMALL), use_edge_features=False)
    assert model.head[0].in_features == 2 * 16
    with pytest.raises(ValueError, match="no input"):
        EdgeClassifier("mlp", 5, 7, 2, GNNConfig(**SMALL), use_edge_features=False)


@pytest.mark.parametrize(
    "ablation", ["none", "shuffled_structure", "topology_only", "no_propagation", "no_self_loops"]
)
def test_every_ablation_runs(tmp_cfg: ExperimentConfig, splits: dict, ablation: str) -> None:
    cfg = with_gnn(
        tmp_cfg,
        models=["host_prior", "mlp", "sage"],
        ablation=ablation,
        protocols=["random", "unseen_pair"],
        tasks=["binary"],
        holdouts=["pair_fold_0_of_5"],
    )
    summary = run_all(cfg, splits)
    assert set(summary["model"]) == set(ablation_models(cfg))
    assert summary["test_f1"].between(0, 1).all()
    pair = summary[summary["protocol"] == "unseen_pair"]
    assert "test_holdout_recall_at_fpr1" in pair.columns
    assert "test_non_self_loop_fpr" in summary.columns


def test_audit_passes_on_clean_graphs(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    host = attacker_hosts(splits, 1)[0]
    for protocol, holdout in (
        ("random", None),
        ("unseen_attack", "mitm"),
        ("unseen_host", host),
        ("unseen_pair", "pair_fold_1_of_5"),
    ):
        result = audit_graph(graph_for(tmp_cfg, splits, protocol, holdout), splits, tmp_cfg)
        failed = [k for k, v in result["checks"].items() if not v["passed"]]
        assert not failed, (protocol, failed)
        assert 0 <= result["exposure"]["test_pair_seen_in_train"] <= 1
    data = graph_for(tmp_cfg, splits, "random", None)
    assert training_ignores_test_labels(data, epochs=1)["passed"]


def test_audit_catches_test_edges_in_message_passing(
    tmp_cfg: ExperimentConfig, splits: dict
) -> None:
    data = graph_for(tmp_cfg, splits, "random", None)
    data.mp_mask = data.mp_mask | data.test_mask
    checks = audit_graph(data, splits, tmp_cfg)["checks"]
    assert not checks["test_not_in_message_passing"]["passed"]


def test_headline_reports_actual_fpr_and_unseen_pair() -> None:
    rows = [
        {
            "protocol": p,
            "task": "binary",
            "model": "m",
            "imbalance": "cw",
            "holdout": h,
            "tuned_for": False,
            "seed": 1,
            "test_f1": 0.9,
            "test_recall_at_fpr1": 0.9,
            "test_fpr_at_fpr1": 0.012,
            "test_holdout_recall_at_fpr1": 0.7,
            "test_holdout_fpr_at_fpr1": 0.03,
        }
        for p, h in (("random", None), ("unseen_pair", "pair_fold_0_of_5"))
    ]
    table = headline_table(pd.DataFrame(rows), [])
    assert table.loc["m", "random actual test FPR@1%"] == "0.0120"
    assert table.loc["m", "unseen pair mean recall@1%FPR"] == "0.7000"
    assert table.loc["m", "unseen pair held-out normal FPR@1%"] == "0.0300"
