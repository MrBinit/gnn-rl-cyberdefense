"""Shortcut references: what host reputation alone achieves, with no flow features.

``host_prior_proba`` scores a flow ``u -> v`` with the class distribution of the training
flows ``u`` sent and of the training flows ``v`` received (the mean of the two when both
exist, the available one otherwise, the overall training distribution when neither host
has a training flow). It only uses **training labels**, which every model is allowed to
learn from, and no features or learning, so any model it matches is likely just
recalling which hosts attack.
"""

import torch
from torch_geometric.utils import scatter


def host_prior_proba(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    n_classes: int,
) -> torch.Tensor:
    """Per-edge class probabilities ``[num_edges, n_classes]`` from endpoint histories."""
    src, dst = edge_index[:, train_mask]
    onehot = torch.nn.functional.one_hot(y[train_mask], n_classes).float()
    sent = scatter(onehot, src, 0, dim_size=num_nodes, reduce="sum")
    received = scatter(onehot, dst, 0, dim_size=num_nodes, reduce="sum")
    prior = onehot.mean(0)

    def dist(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = counts.sum(1, keepdim=True)
        return counts / total.clamp(min=1), total > 0

    p_out, has_out = dist(sent[edge_index[0]])
    p_in, has_in = dist(received[edge_index[1]])
    n = has_out.float() + has_in.float()
    combined = (p_out * has_out + p_in * has_in) / n.clamp(min=1)
    return torch.where(n > 0, combined, prior.expand_as(combined))
