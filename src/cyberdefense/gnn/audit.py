"""Leakage audit for the GNN stage (Phase 2): hard checks plus shortcut-exposure numbers.

Hard checks (any failure = leakage or misalignment):
- ``validate_graph`` on every protocol/holdout graph (docs/04: alignment, labels, masks,
  holdouts absent from train/val/message passing, held-out host isolated, ...);
- ``features_match_baselines``: edge features recomputed independently the baselines' way
  (``make_preprocessor`` fit on the training rows minus the holdout) equal ``edge_attr``;
- ``test_not_in_message_passing``: under train_only, no test (or val) edge is a message-
  passing edge;
- ``node_inputs_ignore_eval_edges``: perturbing every non-message-passing edge (features
  and direction) leaves node inputs and the convolution graph unchanged;
- ``held_out_host_has_no_inputs`` / ``held_out_pair_not_connected``;
- ``training_ignores_test_labels`` (random graph): two trainings, one with the test labels
  shuffled and the test features perturbed, give identical predictions on train and val
  edges, and identical test predictions when only the labels are shuffled.

Exposure (not pass/fail, the shortcuts the audit is looking for): how many test flows have
endpoints and host pairs that also appear in training, how many held-out attack flows
come from a host that already attacks in training (source reputation) or hit a known
victim (destination reputation), how many target flows hit a host never seen in training
versus how often normal test flows have both endpoints unseen (novel-endpoint shortcut),
and what the label-only ``host_prior`` achieves at the fixed FPRs.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from cyberdefense.config import ExperimentConfig, GNNConfig
from cyberdefense.data.preprocess import make_preprocessor
from cyberdefense.data.protocols import holdout_hosts, holdout_mask, target_mask
from cyberdefense.data.ton_iot import SPLIT_COL, SPLITS
from cyberdefense.gnn.models import EdgeClassifier
from cyberdefense.gnn.shortcut import host_prior_proba
from cyberdefense.gnn.train import fit, predict_proba, prepare_inputs
from cyberdefense.graph.build import concat_splits
from cyberdefense.graph.validation import validate_graph
from cyberdefense.metrics import fixed_fpr_metrics


def _check(checks: dict, name: str, ok: bool, detail: object = None) -> None:
    checks[name] = {"passed": bool(ok), **({"detail": detail} if detail is not None else {})}


def _holdout(data: Data) -> str | None:
    return getattr(data, "holdout", None)  # PyG drops attributes set to None


def _held(data: Data, df: pd.DataFrame) -> np.ndarray:
    if _holdout(data) is None:
        return np.zeros(len(df), dtype=bool)
    return holdout_mask(df, data.protocol, _holdout(data))


def audit_graph(data: Data, splits: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> dict:
    """Hard checks and exposure numbers for one protocol/holdout graph."""
    df = concat_splits(splits)
    held = _held(data, df)
    checks = {f"validate_graph:{k}": v for k, v in validate_graph(data, splits)["checks"].items()}

    pre = make_preprocessor(cfg.ton_iot)
    train_rows = (df[SPLIT_COL] == "train").to_numpy() & ~held
    pre.fit(df[train_rows])
    expected = torch.from_numpy(pre.transform(df).astype(np.float32))
    diff = float((expected - data.edge_attr).abs().max())
    _check(checks, "features_match_baselines", diff < 1e-5, {"max_abs_diff": diff})

    mp, test, val = data.mp_mask, data.test_mask, data.val_mask
    if data.message_passing == "train_only":
        leaked = int((mp & (test | val)).sum())
        _check(checks, "test_not_in_message_passing", leaked == 0, leaked)

    before = prepare_inputs(data)
    other = data.clone()
    other.edge_attr = data.edge_attr.clone()
    other.edge_index = data.edge_index.clone()
    other.edge_attr[~mp] += 100.0
    other.edge_index[:, ~mp] = other.edge_index[:, ~mp].flip(0)
    after = prepare_inputs(other)
    _check(
        checks,
        "node_inputs_ignore_eval_edges",
        torch.equal(before.node_in, after.node_in)
        and torch.equal(before.structure, after.structure),
    )

    held_hosts = holdout_hosts(data.protocol, _holdout(data))
    if held_hosts and data.message_passing == "train_only":
        idx = torch.tensor([data.node_ips.index(h) for h in held_hosts])
        touches = int(
            (torch.isin(before.structure[0], idx) | torch.isin(before.structure[1], idx)).sum()
        )
        empty = bool((before.node_in[idx, data.x.shape[1] :] == 0).all())
        _check(checks, "held_out_host_has_no_inputs", empty and touches == 0, touches)
    if data.protocol == "unseen_pair" and data.message_passing == "train_only":
        test_held = data.edge_index[:, torch.from_numpy(held) & test]
        pairs = {tuple(sorted(p)) for p in test_held.t().tolist()}
        mp_pairs = {tuple(sorted(p)) for p in data.edge_index[:, mp].t().tolist()}
        overlap = len(pairs & mp_pairs)
        _check(checks, "held_out_pair_not_connected", overlap == 0, overlap)

    target = target_mask(df, data.protocol, _holdout(data)) if _holdout(data) else held
    return {"checks": checks, "exposure": exposure(data, df, target, cfg)}


def exposure(
    data: Data, df: pd.DataFrame, held: np.ndarray, cfg: ExperimentConfig
) -> dict[str, float]:
    """How much the test flows (and the held-out targets) overlap with what training saw.

    ``held`` is the evaluation target mask (``target_mask``): its attack rows are the target.
    """
    ei = data.edge_index.numpy()
    y = data.y.numpy()
    train = data.train_mask.numpy()
    test = data.test_mask.numpy()
    n = data.num_nodes

    def has(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        seen = np.zeros(n, dtype=bool)
        seen[nodes[mask]] = True
        return seen

    sends = has(ei[0], train)
    receives = has(ei[1], train)
    attacker = has(ei[0], train & (y == 1))
    victim = has(ei[1], train & (y == 1))
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    pair_id = lo.astype(np.int64) * n + hi
    seen_pair = np.isin(pair_id, pair_id[train])

    target = test & (y == 1) & (held if _holdout(data) else True)
    seen = sends | receives  # host has any training flow
    normal_test = test & (y == 0)
    out = {
        "test_flows": int(test.sum()),
        # Novel-endpoint shortcut: "both ends unknown -> attack" is only informative if
        # normal test traffic rarely has both ends unknown.
        "test_normal_both_endpoints_unseen": float(
            (~seen[ei[0][normal_test]] & ~seen[ei[1][normal_test]]).mean()
        ),
        "test_src_sends_in_train": float(sends[ei[0][test]].mean()),
        "test_dst_receives_in_train": float(receives[ei[1][test]].mean()),
        "test_pair_seen_in_train": float(seen_pair[test].mean()),
        "target_attack_flows": int(target.sum()),
    }
    if target.any():
        out |= {
            "target_src_is_known_attacker": float(attacker[ei[0][target]].mean()),
            "target_dst_is_known_victim": float(victim[ei[1][target]].mean()),
            "target_pair_seen_in_train": float(seen_pair[target].mean()),
            "target_dst_unseen_in_train": float((~seen[ei[1][target]]).mean()),
        }
    proba = host_prior_proba(data.edge_index, data.y, data.train_mask, n, 2)[:, 1].numpy()
    val = data.val_mask.numpy()
    _, _, fixed = fixed_fpr_metrics(
        y_val=y[val],
        val_scores=proba[val],
        y_test=y[test],
        test_scores=proba[test],
        fprs=cfg.baseline.fixed_fprs,
        test_holdout=held[test] if _holdout(data) else None,
    )
    prefix = "holdout_recall_at_" if _holdout(data) else "recall_at_"
    out |= {f"host_prior_{k}": v for k, v in fixed.items() if k.startswith((prefix, "fpr_at_"))}
    return out


def training_ignores_test_labels(data: Data, seed: int = 42, epochs: int = 2) -> dict:
    """Train on CPU (one thread, bit-exact); test labels/features must change nothing.

    Passes only if an identical rerun, a run with shuffled test labels (all edges), and a
    run with shuffled test labels plus perturbed test features (train/val edges) give
    exactly the reference predictions.
    """
    cfg = GNNConfig(max_epochs=epochs, patience=epochs, hidden_dim=32, gat_heads=2)
    idx = {s: getattr(data, f"{s}_mask").nonzero().squeeze(1) for s in SPLITS}
    gen = torch.Generator().manual_seed(seed)
    y_shuffled = data.y.clone()
    test = idx["test"]
    y_shuffled[test] = data.y[test][torch.randperm(len(test), generator=gen)]
    perturbed = data.clone()
    perturbed.edge_attr = data.edge_attr.clone()
    perturbed.edge_attr[test] += 100.0

    def train(graph: Data, y: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(seed)
        inputs = prepare_inputs(graph)
        model = EdgeClassifier("sage", inputs.node_in.shape[1], inputs.edge_attr.shape[1], 2, cfg)
        fit(model, inputs, y, idx["train"], idx["val"], 2, cfg, seed)
        return predict_proba(model, inputs, torch.arange(graph.edge_index.shape[1]))

    # Multithreaded CPU kernels are not bit-reproducible (two runs on identical inputs
    # differed by ~5e-3), which would hide or fake a difference. One thread makes every
    # run exact, so any nonzero difference is real influence.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = train(data, data.y)
        rerun = train(data, data.y)
        labels_only = train(data, y_shuffled)
        both = train(perturbed, y_shuffled)
    finally:
        torch.set_num_threads(threads)
    seen = data.train_mask | data.val_mask
    diff_rerun = float((reference - rerun).abs().max())
    diff_labels = float((reference - labels_only).abs().max())
    diff_seen = float((reference[seen] - both[seen]).abs().max())
    return {
        "passed": diff_rerun == 0 and diff_labels == 0 and diff_seen == 0,
        "detail": {
            "max_diff_identical_rerun": diff_rerun,
            "max_diff_all_edges_test_labels_shuffled": diff_labels,
            "max_diff_train_val_edges_test_labels_and_features_changed": diff_seen,
        },
    }
