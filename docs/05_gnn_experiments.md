# 05 · Phase 1 GNN experiments: MLP, GCN, GraphSAGE, GAT

**Status:** Phase 1 run (`results/comparison/gnn.md`). **Reputation-confounded, don't cite as GNN gains:** Phase 2 (docs/06)
shows the improvement comes from per-host traffic profiles, not graph structure (shuffled structure and zero message
passing score the same), and on unseen hosts Random Forest beats every GNN. Phase 3 (docs/07): no generalisation to unseen attacks from unseen hosts either.

Question: do graph relationships improve detection, especially of rare and unseen attacks, at the same false-positive
rate, over a graph-free neural network (MLP) and the frozen Random Forest / XGBoost baselines (`frozen/baselines_v2`)?

## Run
```bash
uv run python scripts/build_graph.py            # once: graph validates (docs/04)
uv run python scripts/train_gnn.py --gnn.device mps   # 400 runs on the Apple GPU (MPS)
uv run python scripts/compare_models.py         # -> results/comparison/gnn.md
```
On an M4 Pro, MPS trains the GNNs 2–4x faster than CPU per epoch (GraphSAGE 0.14 s vs 0.54 s); the MLP is faster on CPU, but it is cheap either way. Run every model on the same device, since MPS and CPU results are not bit-identical. Expect about 1–2 hours on MPS. `--gnn.resume` continues after an interruption. Settings:
`GNNConfig` in `src/cyberdefense/config.py`; code: `src/cyberdefense/gnn/`.

## Models (`cyberdefense.gnn.models`)
Every model scores a flow `u → v` with the **same edge head**, an MLP (hidden 128, two layers, dropout 0.2) on
`[h_u, h_v, edge_features]`. Only the node encoder that produces `h` differs:

| Model | Node encoder | What it can use |
|---|---|---|
| `mlp` | none: the head sees the 102 edge features only | the flow itself (graph-free control, like RF/XGBoost) |
| `gcn` | Linear → 2 × GCNConv | the flow plus its endpoints' neighbourhoods, degree-normalised sum |
| `sage` | Linear → 2 × SAGEConv (mean) | the same, with mean aggregation and a separate self weight |
| `gat` | Linear → 2 × GATConv (4 heads) | the same, with learned attention over neighbours |

**Node input** (identical for the three GNNs, computed from message-passing edges only):
- the node feature `x` (constant 1.0 by default);
- the mean of the 102 features of the flows the host sends, and of the flows it receives;
- two flags for whether it has any.

This carries a host's behaviour, never its identity (no IPs, no learnable per-node embeddings, as CLAUDE.md requires).
A host without message-passing edges gets zeros.

**Convolution graph:** the message-passing flows collapsed to a simple undirected host graph: 1,752 directed edges
(876 pairs), duplicates and self-loops removed. In the flow multigraph a single 15,772-flow pair would dominate
every aggregation (docs/04 §3).

## Training (`cyberdefense.gnn.train`): identical for every model
| Setting | Value |
|---|---|
| Splits | frozen `train`/`val`/`test`, checked with `verify_splits` against `frozen/baselines_v2/manifest.json` |
| Loss | cross-entropy with balanced class weights `n / (k · n_c)` from training edges (binary: normal 2.27, attack 0.64), the same weights as the baselines' `class_weight` |
| Optimiser | Adam, lr 1e-3, weight decay 1e-5, 4,096 training edges per step, node embeddings recomputed each step |
| Early stopping | class-weighted **val** loss, patience 10, max 100 epochs; the best epoch's weights are restored |
| Seeds | 42–46, the same as the baselines |
| Hyperparameters | defaults, **untuned**. Compare against the untuned baselines; tuning must follow docs/03 (holdouts xss/dos/ddos) |

## Evaluation: the baselines' code, so the numbers are like for like
- **Binary threshold:** tuned for F1 on val by `metrics.f1_threshold`. This reproduces the baselines'
  `TunedThresholdClassifierCV` exactly; a test checks equality.
- **Recall at 0.25% and 1% FPR:** thresholds from val normal flows (`metrics.fixed_fpr_metrics`). The headline
  metric, as in docs/02 §4.
- **Held-out detection:** `metrics.holdout_metrics` (now shared with the baselines). It gives recall, ROC-AUC
  against all normal test flows, seen-attack recall and, for hosts, the device's own normal FPR.
- **Multiclass:** macro-F1 and per-class F1.
- **Protocols:** `random` (binary and multiclass), `unseen_attack` (9 types) and `unseen_host` (9 hosts), with
  the same holdouts as the baselines (`cyberdefense.data.protocols`).
- **Per-run JSON:** the same keys as the baselines, plus the epoch history, so `cyberdefense.reporting` builds
  one table across both stages.

## Leakage prevention
1. **Graph checks before every protocol/holdout:** each graph is rebuilt and passes `validate_graph` (docs/04)
   before any model trains on it. A failure stops the run.
2. **Message passing over training edges only** (`train_only`). Node inputs and the convolution graph come from
   `message_passing_graph(data)`. A test confirms this: changing every val/test flow's features and direction
   leaves all node embeddings and predictions unchanged.
3. **Held-out hosts:** under `unseen_host` the host has zero node input and no convolution edges (tested). Its
   test flows are classified from their own features plus the other endpoint.
4. **Held-out attack types** are absent from train, val and message passing. Early stopping and threshold
   tuning use val only; test is touched once, at the end.

## Comparison (`scripts/compare_models.py`)
Reads `frozen/baselines_v2/results/baselines/summary.csv` (add the tuned file with `--gnn.compare_with`) and
`results/gnn/summary.csv`. It writes `results/comparison/gnn.md` with:
- a headline table: random F1 and recall at fixed FPR; multiclass macro-F1; genuine-unseen-attack mean and
  worst recall at fixed FPR (types other than xss/dos/ddos, the same for every model); mean held-out ROC-AUC;
  unseen-host mean and worst recall at fixed FPR;
- every per-holdout table.

**Reference (untuned RF, docs/02):**
- genuine unseen recall: 0.849 at 0.25% FPR, 0.986 at 1%;
- hosts: 0.969 at 0.25%, 0.991 at 1%;
- multiclass macro-F1: 0.957 (XGBoost).

**How to read it:**
- GNN − MLP isolates the graph's contribution, since the head, features, loss and training are identical.
- MLP − RF/XGBoost is the model-family gap.
- Random-split scores are saturated and partly reflect the host shortcut (docs/04 §1). The claim rests on the
  unseen-attack and unseen-host recall at the same FPR.

## Results
_Pending: run the commands above, then fill in from `results/comparison/gnn.md`._

## Known limitations / next steps
- **Untuned GNNs vs tuned baselines** is not a fair comparison. Use the untuned baselines, or tune the GNNs with
  the docs/03 protocol.
- **Node input encodes a host's typical traffic.** That is behaviour, not identity, but on the random split it
  lets the GNNs exploit the source-host shortcut (F1 0.994).
- **Self-loops:** the head can see `h_u = h_v` on a self-loop, and every self-loop flow is normal. If GNN random
  scores look too good, ablate it.
- **Further variants:** `--graph.message_passing transductive` and `--graph.node_features degree` run as
  separate experiments (`results/gnn_transductive`, `results/gnn_degree`). GAT with edge features (`edge_dim`)
  is a later ablation.
