"""Training loop shared by every edge classifier: identical conditions for all models.

- Class-weighted cross-entropy with balanced weights from the training edges (the same
  ``n / (k * n_c)`` as the baselines' ``compute_sample_weight("balanced")``).
- Adam, mini-batches of training edges. Node embeddings are recomputed every step from
  the message-passing graph (776 hosts, so the encoder is cheap).
- Early stopping on the class-weighted validation loss; the best epoch's weights are
  restored. Test edges are never touched during training.
"""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

from cyberdefense.config import Ablation, GNNConfig
from cyberdefense.gnn.models import EdgeClassifier, node_inputs, rewire, structure
from cyberdefense.graph.build import message_passing_graph, node_features


@dataclass
class GraphInputs:
    """Tensors a model needs, on one device. Built from ``mp_mask`` edges only."""

    node_in: torch.Tensor  # node encoder input (models.node_inputs)
    structure: torch.Tensor  # simple undirected message-passing graph (models.structure)
    edge_index: torch.Tensor  # every flow, for classification
    edge_attr: torch.Tensor


def prepare_inputs(
    data: Data, device: str = "cpu", ablation: Ablation = "none", seed: int = 0
) -> GraphInputs:
    """Model inputs from ``mp_mask`` edges only, with the ablation's change applied.

    - topology_only: node input = constant + degree features (no flow features);
    - shuffled_structure: the convolution graph is randomly rewired (``seed``);
    - everything else: the standard inputs.
    """
    mp_index, mp_attr = message_passing_graph(data)
    if ablation == "topology_only":
        degrees, _ = node_features(data.edge_index, data.num_nodes, data.mp_mask, "degree")
        node_in = torch.cat([torch.ones(data.num_nodes, 1), degrees], 1)
    else:
        node_in = node_inputs(data.x, mp_index, mp_attr, data.num_nodes)
    struct = structure(mp_index, data.num_nodes)
    if ablation == "shuffled_structure":
        struct = rewire(struct, data.num_nodes, seed)
    return GraphInputs(
        node_in=node_in.to(device),
        structure=struct.to(device),
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device),
    )


def without_self_loops(data: Data) -> Data:
    """no_self_loops ablation: drop self-loop flows from train, val and message passing.

    Test is unchanged, so the result stays comparable; the self-loop test flows (all
    normal) then show whether the model still handles them.
    """
    out = data.clone()
    loop = data.edge_index[0] == data.edge_index[1]
    out.train_mask = data.train_mask & ~loop
    out.val_mask = data.val_mask & ~loop
    out.mp_mask = data.mp_mask & ~loop
    if data.x.shape[1] > 1:  # degree features must follow the new message-passing edges
        out.x, _ = node_features(data.edge_index, data.num_nodes, out.mp_mask, "degree")
    return out


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Balanced weights ``n / (k * n_c)`` over the k classes present; absent classes get 0."""
    counts = torch.bincount(y, minlength=n_classes).float()
    present = counts > 0
    weights = torch.zeros(n_classes)
    weights[present] = len(y) / (int(present.sum()) * counts[present])
    return weights


@dataclass
class FitResult:
    best_epoch: int
    epochs_run: int
    best_val_loss: float
    fit_seconds: float
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def predict_proba(
    model: EdgeClassifier, inputs: GraphInputs, idx: torch.Tensor, batch_size: int = 65536
) -> torch.Tensor:
    """Class probabilities for the edges ``idx`` (eval mode, no dropout)."""
    model.eval()
    h = model.encode(inputs.node_in, inputs.structure)
    out = [
        torch.softmax(model.classify(h, inputs.edge_index[:, b], inputs.edge_attr[b]), dim=1)
        for b in idx.split(batch_size)
    ]
    return torch.cat(out).cpu()


@torch.no_grad()
def _loss(
    model: EdgeClassifier,
    inputs: GraphInputs,
    y: torch.Tensor,
    idx: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> tuple[float, np.ndarray]:
    model.eval()
    h = model.encode(inputs.node_in, inputs.structure)
    logits = model.classify(h, inputs.edge_index[:, idx], inputs.edge_attr[idx])
    return float(loss_fn(logits, y[idx])), logits.argmax(1).cpu().numpy()


def fit(
    model: EdgeClassifier,
    inputs: GraphInputs,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    n_classes: int,
    cfg: GNNConfig,
    seed: int,
    writer: SummaryWriter | None = None,
) -> FitResult:
    """Train with early stopping on val loss; leaves ``model`` at its best epoch."""
    device = inputs.edge_attr.device
    y = y.to(device)
    weights = class_weights(y[train_idx].cpu(), n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    average = "binary" if n_classes == 2 else "macro"

    best, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history, start = [], time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).to(train_idx.device)]
        total = 0.0
        for b in perm.split(cfg.batch_size):
            h = model.encode(inputs.node_in, inputs.structure)
            logits = model.classify(h, inputs.edge_index[:, b], inputs.edge_attr[b])
            loss = loss_fn(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(train_idx)
        val_loss, val_pred = _loss(model, inputs, y, val_idx, loss_fn)
        val_f1 = f1_score(y[val_idx].cpu().numpy(), val_pred, average=average, zero_division=0)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("f1_argmax/val", val_f1, epoch)

        if val_loss < best - cfg.min_delta:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return FitResult(
        best_epoch=best_epoch,
        epochs_run=len(history),
        best_val_loss=best,
        fit_seconds=time.perf_counter() - start,
        history=history,
    )
