"""Host-graph construction and validation."""

import numpy as np
import pandas as pd
import pytest
import torch

from cyberdefense.config import ExperimentConfig, GraphConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.ton_iot import SPLITS, load_split
from cyberdefense.graph.build import build_graph, load_graph, message_passing_graph, save_graph
from cyberdefense.graph.validation import describe_graph, validate_graph, write_figures


@pytest.fixture
def splits(tmp_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    cfg = tmp_cfg.model_copy(
        update={"ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    return {name: load_split(cfg.ton_iot, name) for name in SPLITS}


def with_graph(cfg: ExperimentConfig, **kwargs) -> ExperimentConfig:
    return cfg.model_copy(update={"graph": GraphConfig(**kwargs)})


def test_graph_matches_splits(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    data = build_graph(splits, tmp_cfg)
    n_rows = sum(len(df) for df in splits.values())
    hosts = set().union(*(set(df["src_ip"]) | set(df["dst_ip"]) for df in splits.values()))

    assert data.edge_index.shape == (2, n_rows)
    assert data.num_nodes == len(hosts)
    assert data.x.shape == (len(hosts), 1)  # constant: no identity
    assert data.edge_attr.shape == (n_rows, len(data.edge_feature_names))
    assert int(data.train_mask.sum()) == len(splits["train"])
    assert torch.equal(data.mp_mask, data.train_mask)  # train_only by default
    first = splits["train"].iloc[0]
    assert data.node_ips[int(data.edge_index[0, 0])] == first["src_ip"]
    assert data.node_ips[int(data.edge_index[1, 0])] == first["dst_ip"]

    report = validate_graph(data, splits)
    assert report["passed"], report["hard_failures"]


def test_unseen_host_leaves_training_graph(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    host = splits["train"].loc[splits["train"]["label"] == 1, "src_ip"].value_counts().index[0]
    data = build_graph(splits, with_graph(tmp_cfg, protocol="unseen_host", holdout=host))
    idx = data.node_ips.index(host)
    mp_index, mp_attr = message_passing_graph(data)
    assert not ((mp_index[0] == idx) | (mp_index[1] == idx)).any()
    assert mp_attr.shape[0] == mp_index.shape[1]
    test = splits["test"]
    in_test = int(((test["src_ip"] == host) | (test["dst_ip"] == host)).sum())
    assert int(data.test_mask.sum()) == len(test)  # its test flows are still evaluated
    assert in_test > 0
    assert validate_graph(data, splits)["passed"]


def test_unseen_attack_and_transductive(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    cfg = with_graph(
        tmp_cfg, protocol="unseen_attack", holdout="mitm", message_passing="transductive"
    )
    data = build_graph(splits, cfg)
    mitm = data.class_names.index("mitm")
    assert not (data.y_multi[data.train_mask | data.val_mask] == mitm).any()
    # Transductive: test mitm flows are visible structure (features, not labels).
    assert (data.y_multi[data.mp_mask] == mitm).any()
    assert validate_graph(data, splits)["passed"]


def test_degree_features_use_message_passing_edges_only(
    tmp_cfg: ExperimentConfig, splits: dict
) -> None:
    data = build_graph(splits, with_graph(tmp_cfg, node_features="degree"))
    out_flows = torch.bincount(data.edge_index[0, data.train_mask], minlength=data.num_nodes)
    assert torch.allclose(data.x[:, 0], torch.log1p(out_flows.float()))
    assert validate_graph(data, splits)["passed"]


def test_validation_catches_corruption(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    data = build_graph(splits, tmp_cfg)
    data.val_mask[0] = True  # a train edge also in val
    data.y[1] = 1 - data.y[1]
    data.edge_attr[2, 0] = float("nan")
    failures = validate_graph(data, splits)["hard_failures"]
    assert {"split_masks_disjoint", "binary_labels_match", "edge_features_finite"} <= set(failures)


def test_holdout_config_guard(tmp_cfg: ExperimentConfig, splits: dict) -> None:
    with pytest.raises(ValueError, match="holdout"):
        build_graph(splits, with_graph(tmp_cfg, protocol="unseen_host"))


def test_describe_save_and_figures(tmp_cfg: ExperimentConfig, splits: dict, tmp_path) -> None:
    data = build_graph(splits, tmp_cfg)
    stats = describe_graph(data)
    assert stats["edges"] == data.edge_index.shape[1]
    assert sum(stats["connected_components"]) == data.num_nodes
    assert 0 <= stats["source_host_shortcut_test_f1"] <= 1

    save_graph(data, tmp_path / "g.pt")
    loaded = load_graph(tmp_path / "g.pt")
    assert torch.equal(loaded.edge_index, data.edge_index)
    assert loaded.node_ips == data.node_ips
    assert all(p.exists() for p in write_figures(data, tmp_path / "fig"))
    assert np.isfinite(loaded.edge_attr.numpy()).all()
