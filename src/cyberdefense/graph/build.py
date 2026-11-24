"""Homogeneous host graph for edge classification (PyTorch Geometric).

- **Nodes** are hosts: every IP address seen as a source or destination in any split, in
  sorted order so node indices are deterministic. IPs are kept in ``data.node_ips`` for
  analysis only and are never features.
- **Edges** are flows: one directed edge ``src_ip -> dst_ip`` per row, so the graph is a
  multigraph (many flows per host pair) with self-loops where a host talks to itself.
  Edge order is train rows, then val, then test, exactly as stored in the split files.
- **Edge features** are the baselines' features (``cyberdefense.data.preprocess``), with the
  preprocessor fitted on this graph's training edges only, as ``fit_baseline`` does.
- **Labels** are per edge: ``y`` (1 = attack) and ``y_multi`` (index into ``class_names``).
- **Masks** (per edge): ``train_mask`` / ``val_mask`` / ``test_mask`` say which edges are
  trained on and evaluated; ``mp_mask`` says which edges a GNN may propagate messages over.
  Held-out edges (unseen_attack / unseen_host protocols) are removed from train, val and
  message passing, exactly the rows the baselines drop.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from cyberdefense.config import ExperimentConfig, GraphConfig
from cyberdefense.data.preprocess import make_preprocessor
from cyberdefense.data.protocols import holdout_mask
from cyberdefense.data.schema import SPLIT_COL, SPLITS, TARGET_BINARY, TARGET_MULTI

DEGREE_FEATURES = ["log_out_flows", "log_in_flows", "log_out_neighbours", "log_in_neighbours"]


def _tensor(a: np.ndarray) -> torch.Tensor:
    """Owned copy: pandas/Arrow-backed arrays can be read-only, which torch must not share."""
    return torch.from_numpy(np.array(a, copy=True))


def concat_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Train, val and test rows in that order with a ``split`` column (one row per edge)."""
    if missing := set(SPLITS) - set(splits):
        raise ValueError(f"missing splits: {sorted(missing)}")
    return pd.concat([splits[s].assign(**{SPLIT_COL: s}) for s in SPLITS], ignore_index=True)


def node_features(
    edge_index: torch.Tensor, num_nodes: int, mp_mask: torch.Tensor, kind: str
) -> tuple[torch.Tensor, list[str]]:
    """Node features without identity: a constant, or degrees over message-passing edges."""
    if kind == "constant":
        return torch.ones(num_nodes, 1), ["constant"]
    src, dst = edge_index[:, mp_mask]
    out_flows = torch.bincount(src, minlength=num_nodes)
    in_flows = torch.bincount(dst, minlength=num_nodes)
    pairs = torch.unique(torch.stack([src, dst]), dim=1)
    out_nbrs = torch.bincount(pairs[0], minlength=num_nodes)
    in_nbrs = torch.bincount(pairs[1], minlength=num_nodes)
    x = torch.stack([out_flows, in_flows, out_nbrs, in_nbrs], dim=1).float()
    return torch.log1p(x), DEGREE_FEATURES


def build_graph(splits: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> Data:
    """Build the host graph for ``cfg.graph.protocol`` / ``cfg.graph.holdout``."""
    g: GraphConfig = cfg.graph
    if (g.protocol == "random") != (g.holdout is None):
        raise ValueError("the random protocol takes no holdout; the others need one")
    df = concat_splits(splits)

    node_ips = np.sort(pd.unique(pd.concat([df["src_ip"], df["dst_ip"]]).astype(str)))
    src = pd.Categorical(df["src_ip"].astype(str), categories=node_ips).codes
    dst = pd.Categorical(df["dst_ip"].astype(str), categories=node_ips).codes
    edge_index = _tensor(np.stack([src, dst]).astype(np.int64))

    held = holdout_mask(df, g.protocol, g.holdout) if g.holdout else np.zeros(len(df), bool)
    in_split = {s: (df[SPLIT_COL] == s).to_numpy() for s in SPLITS}
    train = in_split["train"] & ~held
    val = in_split["val"] & ~held
    test = in_split["test"]
    mp = train if g.message_passing == "train_only" else train | val | test

    pre = make_preprocessor(cfg.dataset)
    pre.fit(df[train])
    edge_attr = _tensor(pre.transform(df).astype(np.float32))

    # Same class list as the baselines: every type present in the (full) training split.
    class_names = sorted(splits["train"][TARGET_MULTI].astype(str).unique())
    y_multi = pd.Categorical(df[TARGET_MULTI].astype(str), categories=class_names).codes
    if (y_multi < 0).any():
        raise ValueError("found an attack type that is not in the training class list")

    mp_mask = _tensor(mp)
    x, x_names = node_features(edge_index, len(node_ips), mp_mask, g.node_features)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=_tensor(df[TARGET_BINARY].to_numpy(np.int64)),
        y_multi=_tensor(y_multi.astype(np.int64)),
        train_mask=_tensor(train),
        val_mask=_tensor(val),
        test_mask=_tensor(test),
        mp_mask=mp_mask,
        num_nodes=len(node_ips),
        node_ips=node_ips.tolist(),
        class_names=class_names,
        edge_feature_names=pre.get_feature_names_out().tolist(),
        node_feature_names=x_names,
        protocol=g.protocol,
        holdout=g.holdout,
        message_passing=g.message_passing,
    )


def message_passing_graph(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """``(edge_index, edge_attr)`` restricted to the edges a GNN may propagate over."""
    return data.edge_index[:, data.mp_mask], data.edge_attr[data.mp_mask]


def save_graph(data: Data, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_graph(path) -> Data:
    # The file holds a PyG Data object (not just tensors), so full unpickling is needed;
    # only load graphs this project wrote.
    return torch.load(path, weights_only=False)
