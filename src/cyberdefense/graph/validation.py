"""Graph validation: hard checks that the graph is a faithful, leak-free view of the splits.

Every expectation is recomputed from the split DataFrames, not from the graph itself, so
a bug in ``build_graph`` cannot validate itself. ``describe_graph`` adds the statistics
that explain what the graph looks like; ``write_figures`` draws it.
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from cyberdefense.data.protocols import holdout_hosts, holdout_mask
from cyberdefense.data.schema import (
    SPLIT_COL,
    SPLITS,
    TARGET_BINARY,
    TARGET_MULTI,
)
from cyberdefense.graph.build import concat_splits, node_features

ID_COLUMNS = ("src_ip", "dst_ip", "src_port", "dst_port")


def _check(checks: dict, name: str, ok: bool, detail: object = None) -> None:
    checks[name] = {"passed": bool(ok), **({"detail": detail} if detail is not None else {})}


def validate_graph(data: Data, splits: dict[str, pd.DataFrame]) -> dict:
    """Run the hard checks; ``report["passed"]`` is False if any of them fails."""
    df = concat_splits(splits)
    checks: dict = {}
    n_edges, n_nodes = len(df), data.num_nodes
    ei = data.edge_index

    # Structure: one edge per flow, one node per host, every index valid.
    hosts = np.sort(pd.unique(pd.concat([df["src_ip"], df["dst_ip"]]).astype(str)))
    _check(checks, "one_edge_per_flow", ei.shape == (2, n_edges), list(ei.shape))
    _check(checks, "one_node_per_host", n_nodes == len(hosts) == len(data.node_ips))
    _check(checks, "node_order_sorted_unique", list(data.node_ips) == hosts.tolist())
    _check(
        checks,
        "edge_index_valid",
        ei.dtype == torch.long and int(ei.min()) >= 0 and int(ei.max()) < n_nodes,
    )
    degree = torch.bincount(ei.flatten(), minlength=n_nodes)
    _check(checks, "no_isolated_nodes", int((degree == 0).sum()) == 0, int((degree == 0).sum()))

    # Alignment: edge i is row i of train+val+test (endpoints and labels).
    ips = np.asarray(data.node_ips)
    _check(
        checks,
        "edges_aligned_with_rows",
        bool(
            (ips[ei[0].numpy()] == df["src_ip"].astype(str).to_numpy()).all()
            and (ips[ei[1].numpy()] == df["dst_ip"].astype(str).to_numpy()).all()
        ),
    )
    _check(checks, "binary_labels_match", bool((data.y.numpy() == df[TARGET_BINARY]).all()))
    names = np.asarray(data.class_names)
    _check(
        checks,
        "multiclass_labels_match",
        bool((names[data.y_multi.numpy()] == df[TARGET_MULTI].astype(str).to_numpy()).all()),
    )

    # Masks: disjoint, the right sizes, holdouts removed from train/val/message passing.
    held = (
        holdout_mask(df, data.protocol, getattr(data, "holdout", None))
        if getattr(data, "holdout", None)
        else np.zeros(n_edges, dtype=bool)
    )
    masks = {s: getattr(data, f"{s}_mask").numpy() for s in SPLITS}
    overlap = sum(int((masks[a] & masks[b]).sum()) for a, b in combinations(SPLITS, 2))
    _check(checks, "split_masks_disjoint", overlap == 0, overlap)
    expected = {
        s: int(((df[SPLIT_COL] == s).to_numpy() & (~held if s != "test" else True)).sum())
        for s in SPLITS
    }
    actual = {s: int(m.sum()) for s, m in masks.items()}
    _check(checks, "split_mask_sizes", expected == actual, {"expected": expected, "actual": actual})
    mp = data.mp_mask.numpy()
    # Transductive message passing may see held-out *test* flows (features only, never
    # labels); held-out train/val rows must be gone from everything.
    mp_forbidden = held if data.message_passing == "train_only" else held & ~masks["test"]
    leaked = {
        "train": int((masks["train"] & held).sum()),
        "val": int((masks["val"] & held).sum()),
        "message_passing": int((mp & mp_forbidden).sum()),
    }
    _check(checks, "holdout_absent_from_training_graph", not any(leaked.values()), leaked)
    if data.message_passing == "train_only":
        _check(checks, "message_passing_is_train_edges", bool((mp == masks["train"]).all()))
    else:
        allowed = masks["train"] | masks["val"] | masks["test"]
        _check(checks, "message_passing_within_split_edges", bool((mp <= allowed).all()))
    held_hosts = holdout_hosts(data.protocol, getattr(data, "holdout", None))
    if held_hosts and data.message_passing == "train_only":
        # unseen_host (one host) and unseen_attack_host (every sender of the type): each
        # held-out host must be isolated in the message-passing graph.
        idx = torch.tensor([data.node_ips.index(h) for h in held_hosts])
        mp_ei = ei[:, data.mp_mask]
        touches = int((torch.isin(mp_ei[0], idx) | torch.isin(mp_ei[1], idx)).sum())
        _check(checks, "held_out_host_has_no_training_edges", touches == 0, touches)

    # Features: finite, no identifiers, standardised on the training edges they were fit on.
    attr = data.edge_attr.numpy()
    _check(checks, "edge_features_finite", bool(np.isfinite(attr).all()))
    _check(checks, "edge_feature_names_match", attr.shape[1] == len(data.edge_feature_names))
    # Output names are "num__<col>" or "cat__<col>_<value>" (src_ip_bytes is a real feature).
    id_feats = [
        f
        for f in data.edge_feature_names
        for c in ID_COLUMNS
        if f == f"num__{c}" or f.startswith(f"cat__{c}_")
    ]
    _check(checks, "no_identifier_features", not id_feats, id_feats)
    n_num = sum(name.startswith("num__") for name in data.edge_feature_names)
    num = attr[masks["train"], :n_num].astype(np.float64)
    std = num.std(0)
    varying = std > 0  # a column constant in train is scaled to 0 and carries no information
    worst_mean = float(np.abs(num.mean(0)).max()) if n_num else 0.0
    worst_std = float(np.abs(std[varying] - 1).max()) if varying.any() else 0.0
    _check(
        checks,
        "preprocessor_fit_on_train_edges",
        worst_mean < 1e-3 and worst_std < 1e-2,
        {"max_abs_mean": worst_mean, "max_abs_std_minus_1": worst_std},
    )
    x = data.x
    if data.node_feature_names == ["constant"]:
        _check(checks, "node_features_carry_no_identity", bool((x == 1).all()))
    else:
        recomputed, _ = node_features(ei, n_nodes, data.mp_mask, "degree")
        _check(
            checks,
            "node_features_from_message_passing_edges",
            bool(torch.allclose(x, recomputed)),
        )

    failures = [name for name, c in checks.items() if not c["passed"]]
    return {"passed": not failures, "hard_failures": failures, "checks": checks}


def describe_graph(data: Data) -> dict:
    """What the graph looks like: size, multiplicity, components, degrees, hubs, splits."""
    ei = data.edge_index.numpy()
    y = data.y.numpy()
    ips = np.asarray(data.node_ips)
    n = data.num_nodes
    edges = pd.DataFrame({"src": ei[0], "dst": ei[1], "y": y})
    pairs = edges.groupby(["src", "dst"]).size()
    undirected = nx.Graph()
    undirected.add_nodes_from(range(n))
    undirected.add_edges_from(pairs.index)
    components = sorted((len(c) for c in nx.connected_components(undirected)), reverse=True)

    flows = np.bincount(ei.ravel(), minlength=n)
    out_flows = np.bincount(ei[0], minlength=n)
    nbrs = np.array([undirected.degree(i) for i in range(n)])
    attack_flows = np.bincount(ei.ravel(), weights=np.r_[y, y], minlength=n)
    hubs = np.argsort(-flows)[:10]

    masks = {s: getattr(data, f"{s}_mask").numpy() for s in SPLITS}
    train_nodes = np.zeros(n, dtype=bool)
    train_nodes[ei[:, masks["train"]].ravel()] = True
    per_split = {}
    for s, m in masks.items():
        touched = np.unique(ei[:, m].ravel())
        per_split[s] = {
            "edges": int(m.sum()),
            "attack_share": float(y[m].mean()) if m.any() else None,
            "hosts": len(touched),
            "hosts_without_training_edges": int((~train_nodes[touched]).sum()),
            "edges_touching_such_hosts": int(
                (m & ~(train_nodes[ei[0]] & train_nodes[ei[1]])).sum()
            ),
        }

    # How much the source host alone gives away: majority train label of the src host.
    tr = edges[masks["train"]]
    host_rate = tr.groupby("src")["y"].mean()
    te = edges[masks["test"]]
    guess = te["src"].map(host_rate).fillna(1.0).ge(0.5).astype(int)
    tp = int(((guess == 1) & (te["y"] == 1)).sum())
    shortcut_f1 = 2 * tp / max(int(guess.sum()) + int(te["y"].sum()), 1)

    self_loops = ei[0] == ei[1]
    return {
        "nodes": n,
        "edges": int(ei.shape[1]),
        "message_passing_edges": int(data.mp_mask.sum()),
        "edge_features": int(data.edge_attr.shape[1]),
        "node_features": list(data.node_feature_names),
        "source_hosts": int((out_flows > 0).sum()),
        "destination_only_hosts": int(((out_flows == 0) & (flows > 0)).sum()),
        "distinct_directed_pairs": len(pairs),
        "distinct_undirected_pairs": undirected.number_of_edges(),
        "flows_per_pair": {
            "median": float(pairs.median()),
            "mean": float(pairs.mean()),
            "max": int(pairs.max()),
            "pairs_with_one_flow": int((pairs == 1).sum()),
        },
        "self_loop_edges": int(self_loops.sum()),
        "self_loop_attack_share": float(y[self_loops].mean()) if self_loops.any() else None,
        "connected_components": components,
        "density_simple_graph": nx.density(undirected),
        "neighbours_per_host": pd.Series(nbrs).describe().round(2).to_dict(),
        "hosts_with_one_neighbour": int((nbrs == 1).sum()),
        "top_hosts_by_flows": [
            {
                "ip": str(ips[i]),
                "flows": int(flows[i]),
                "sent": int(out_flows[i]),
                "neighbours": int(nbrs[i]),
                "attack_share": round(float(attack_flows[i] / flows[i]), 4),
            }
            for i in hubs
        ],
        "splits": per_split,
        "source_host_shortcut_test_f1": round(shortcut_f1, 4),
    }


def write_figures(data: Data, out_dir: Path) -> list[Path]:
    """Host-graph drawing (multi-edges collapsed) and degree/multiplicity distributions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ei = data.edge_index.numpy()
    y = data.y.numpy()
    n = data.num_nodes
    flows = np.bincount(ei.ravel(), minlength=n)
    attack = np.bincount(ei.ravel(), weights=np.r_[y, y], minlength=n) / np.maximum(flows, 1)
    pairs = pd.DataFrame({"a": ei.min(0), "b": ei.max(0)}).value_counts()

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from((a, b, w) for (a, b), w in pairs.items() if a != b)
    pos = nx.spring_layout(g, seed=42, k=0.6 / np.sqrt(n), iterations=200)
    fig, ax = plt.subplots(figsize=(13, 11))
    widths = [0.3 + 0.6 * np.log10(d["weight"]) for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.25, edge_color="grey")
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=8 + 30 * np.log1p(flows),
        node_color=attack,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        edgecolors="black",
    )
    fig.colorbar(nodes, ax=ax, shrink=0.6, label="attack share of the host's flows")
    ax.set_title(
        f"Host graph: {n} hosts, {ei.shape[1]:,} flows collapsed to "
        f"{g.number_of_edges():,} host pairs\n"
        "node size = log(flows), edge width = log(flows between the pair); self-loops hidden"
    )
    ax.axis("off")
    graph_png = out_dir / "host_graph.png"
    fig.tight_layout()
    fig.savefig(graph_png, dpi=130)
    plt.close(fig)

    nbrs = np.array([g.degree(i) for i in range(n)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(nbrs, bins=np.logspace(0, np.log10(nbrs.max() + 1), 30))
    axes[0].set(
        xscale="log",
        yscale="log",
        title="Distinct neighbours per host",
        xlabel="neighbours",
        ylabel="hosts",
    )
    axes[1].hist(flows, bins=np.logspace(0, np.log10(flows.max() + 1), 30))
    axes[1].set(xscale="log", yscale="log", title="Flows per host", xlabel="flows", ylabel="hosts")
    axes[2].hist(pairs.to_numpy(), bins=np.logspace(0, np.log10(pairs.max() + 1), 30))
    axes[2].set(
        xscale="log", yscale="log", title="Flows per host pair", xlabel="flows", ylabel="pairs"
    )
    deg_png = out_dir / "degree_distributions.png"
    fig.tight_layout()
    fig.savefig(deg_png, dpi=130)
    plt.close(fig)
    return [graph_png, deg_png]


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
