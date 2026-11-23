"""Edge classifiers: one shared head, four node encoders.

Every model scores a flow ``u -> v`` with the same head, an MLP on
``[h_u, h_v, edge_features]`` (the ``mlp`` control drops ``h_u, h_v``). Only the node
encoder that produces ``h`` differs, so differences between models come from how they use
the graph, not from the classifier.

Node encoder input (``node_inputs``), computed from **message-passing edges only**:
the node's own features ``x`` (constant or degree, see GraphConfig), the mean features
of the flows it sends and of the flows it receives, and two flags for whether it has any.
Hosts give no identity: a host's input only describes its (allowed) traffic, so a host
without message-passing edges (e.g. the held-out host under unseen_host) is all zeros.

Convolutions run on ``structure(...)``: the message-passing graph as a simple undirected
graph (one edge per communicating pair, self-loops dropped; the convs add their own). The
flow multigraph would let a single 15,772-flow pair dominate every aggregation.
"""

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import remove_self_loops, scatter, to_undirected

from cyberdefense.config import GNNConfig, GNNModel


def node_inputs(
    x: torch.Tensor, mp_edge_index: torch.Tensor, mp_edge_attr: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """``[x, mean out-flow features, mean in-flow features, has_out, has_in]`` per node."""
    src, dst = mp_edge_index
    ones = mp_edge_attr.new_ones(src.numel(), 1)
    out_mean = scatter(mp_edge_attr, src, 0, dim_size=num_nodes, reduce="mean")
    in_mean = scatter(mp_edge_attr, dst, 0, dim_size=num_nodes, reduce="mean")
    has_out = (scatter(ones, src, 0, dim_size=num_nodes, reduce="sum") > 0).float()
    has_in = (scatter(ones, dst, 0, dim_size=num_nodes, reduce="sum") > 0).float()
    return torch.cat([x, out_mean, in_mean, has_out, has_in], dim=1)


def structure(mp_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Simple undirected graph of the communicating host pairs (deduplicated, no self-loops)."""
    edge_index, _ = remove_self_loops(mp_edge_index)
    return to_undirected(edge_index, num_nodes=num_nodes)


def rewire(structure_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    """Randomly rewired copy of an undirected structure (shuffled_structure ablation).

    Each undirected pair keeps one endpoint and gets a random other endpoint drawn by
    shuffling the second endpoints of all pairs (a configuration-model shuffle): every
    node keeps roughly its degree, but who it is connected to becomes random.
    """
    src, dst = structure_index
    half = src < dst
    a, b = src[half], dst[half]
    gen = torch.Generator().manual_seed(seed)
    b = b[torch.randperm(b.numel(), generator=gen).to(b.device)]
    edge_index, _ = remove_self_loops(torch.stack([a, b]))
    return to_undirected(edge_index, num_nodes=num_nodes)


def _conv(kind: GNNModel, dim: int, heads: int) -> nn.Module:
    if kind == "gcn":
        return GCNConv(dim, dim)
    if kind == "sage":
        return SAGEConv(dim, dim, aggr="mean")
    if kind == "gat":
        return GATConv(dim, dim // heads, heads=heads)
    raise ValueError(f"no convolution for model {kind!r}")


class NodeEncoder(nn.Module):
    """Linear input projection, then ``num_layers`` convolutions with ReLU and dropout.

    ``num_layers=0`` (the no_propagation ablation) keeps only the projection: each node
    embedding then comes from its own inputs, with no neighbour information.
    """

    def __init__(self, kind: GNNModel, in_dim: int, cfg: GNNConfig, num_layers: int) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU())
        self.convs = nn.ModuleList(
            _conv(kind, cfg.hidden_dim, cfg.gat_heads) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.inp(h))
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, edge_index)))
        return h


class EdgeClassifier(nn.Module):
    """Shared edge head on top of an optional node encoder (``mlp`` has none).

    ``use_edge_features=False`` (topology_only) drops the flow's own features from the head,
    which then sees only ``[h_u, h_v]``. ``num_layers`` overrides ``cfg.num_layers``.
    """

    def __init__(
        self,
        kind: GNNModel,
        node_dim: int,
        edge_dim: int,
        n_out: int,
        cfg: GNNConfig,
        *,
        use_edge_features: bool = True,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        if kind == "host_prior":
            raise ValueError("host_prior is not a neural model (see gnn.shortcut)")
        if kind == "mlp" and not use_edge_features:
            raise ValueError("an mlp without edge features has no input")
        self.kind = kind
        self.use_edge_features = use_edge_features
        layers = cfg.num_layers if num_layers is None else num_layers
        self.encoder = None if kind == "mlp" else NodeEncoder(kind, node_dim, cfg, layers)
        head_in = (edge_dim if use_edge_features else 0) + (
            0 if self.encoder is None else 2 * cfg.hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, n_out),
        )

    def encode(self, node_in: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor | None:
        """Node embeddings from the message-passing structure (None for ``mlp``)."""
        return None if self.encoder is None else self.encoder(node_in, edge_index)

    def classify(
        self, h: torch.Tensor | None, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Logits for the flows ``edge_index`` (any split) with features ``edge_attr``."""
        parts = [] if h is None else [h[edge_index[0]], h[edge_index[1]]]
        if self.use_edge_features:
            parts.append(edge_attr)
        return self.head(torch.cat(parts, 1))
