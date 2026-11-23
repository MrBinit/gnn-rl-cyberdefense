# 04 · Graph construction and validation (homogeneous host graph)

How the ToN-IoT flows become a PyTorch Geometric graph for the GNN stage, how the graph is validated and what
it looks like.

## Run
```bash
uv run python scripts/build_graph.py                                # random protocol, validated
uv run python scripts/build_graph.py --graph.node_features degree   # structural node features
uv run python scripts/build_graph.py --graph.protocol unseen_host --graph.holdout 192.168.1.30
```

**Outputs:**
- `data/processed/graph/<protocol>[__<holdout>].pt`: a `torch_geometric.data.Data` object. Load it with
  `cyberdefense.graph.build.load_graph`.
- `results/graph/<name>/report.json`: every validation check, plus the statistics below.
- `results/graph/<name>/host_graph.png` and `degree_distributions.png`.

Settings live in `GraphConfig` (`src/cyberdefense/config.py`); the code is in `src/cyberdefense/graph/`. The
build takes about 45 s, including validating all 18 holdout graphs.

## How the graph is built (`cyberdefense.graph.build.build_graph`)
| Element | Definition | Why |
|---|---|---|
| **Node** | One per host: every IP seen as source or destination in any split (776). Sorted by IP, so indices are deterministic. | The project plan: a host graph with flows as edges. |
| **Edge** | One directed edge `src_ip → dst_ip` per flow (190,444). Order: train rows, then val, then test, as in the split files. | Edge classification needs one labelled edge per flow. The graph is a **multigraph**: one host pair can carry up to 15,772 flows. Self-loops are kept. |
| **Edge features** `edge_attr` | The baselines' 102 features (`cyberdefense.data.preprocess`: log1p + standardise numeric, one-hot categorical). The preprocessor is fit on this graph's **training edges only**. | Identical inputs to RF/XGBoost, so a GNN's gain can only come from the graph structure. |
| **Node features** `x` | Default `constant`: a single 1.0 per node. Option `degree`: log1p in/out flow counts and distinct neighbours, counted on message-passing edges only. | No identity. CLAUDE.md forbids IPs, ports and per-node learnable embeddings. With constant features, everything a node "knows" comes from its neighbours' edge features (the E-GraphSAGE approach). |
| **Labels** | `y`: 1 = attack; `y_multi`: index into `class_names` (the same 10 classes as the baselines). | Binary and multiclass edge classification. |
| **Masks** (per edge) | `train_mask`, `val_mask`, `test_mask`: which edges are trained on and evaluated. `mp_mask`: which edges a GNN may pass messages over. | Keeps the frozen splits and makes leakage explicit. |
| **Metadata** | `node_ips` (analysis only, never a feature), `edge_feature_names`, `node_feature_names`, `protocol`, `holdout`, `message_passing`. | Traceability. |

### Message passing (`--graph.message_passing`)
- **`train_only` (default, conservative):** the GNN propagates over **training edges only**, in every phase.
  Val and test edges are only classified, from their own features and their endpoints' embeddings. A test
  flow's features and label never influence any embedding.
- **`transductive`:** the GNN may propagate over all edges' **features**, never their labels. This matches a
  deployment that sees all traffic but knows no labels. Held-out *test* flows become visible structure, so use
  it as a separate, clearly labelled experiment.

`message_passing_graph(data)` returns `(edge_index, edge_attr)` restricted to `mp_mask`. GNN layers should take
their input from it, never from the full `data.edge_index`.

### Protocols and holdouts
The same holdouts as the baselines (`cyberdefense.data.protocols`, shared code):

| Protocol | What is removed from train, val and message passing | Evaluated on |
|---|---|---|
| `random` | nothing | all test edges |
| `unseen_attack` (9 types) | every edge of that attack type | all test edges; the held-out type is the target |
| `unseen_host` (9 attacker hosts) | every edge the host sends **or receives** | all test edges; the host's edges are the target |

The held-out host stays a node, because its test edges need endpoints. Under `train_only` it has **no edges in
the message-passing graph**, so the model knows nothing about it, as CLAUDE.md requires. The preprocessor is
refit without the held-out edges, as in `fit_baseline`.

## Validation (`cyberdefense.graph.validation.validate_graph`)
Every expectation is recomputed from the split files, not read from the graph, so a builder bug can't validate
itself. A failure exits with status 1.

| Check | What it guarantees |
|---|---|
| frozen manifest (`verify_splits`) | the splits are byte-identical to the `frozen/baselines_v2` baselines |
| `one_edge_per_flow`, `one_node_per_host`, `node_order_sorted_unique`, `edge_index_valid`, `no_isolated_nodes` | the structure is complete and well-formed |
| `edges_aligned_with_rows` | edge *i*'s endpoints are row *i*'s `src_ip` / `dst_ip` |
| `binary_labels_match`, `multiclass_labels_match` | edge labels are the flow labels |
| `split_masks_disjoint`, `split_mask_sizes` | no edge is in two splits; mask sizes equal the split sizes minus holdouts |
| `holdout_absent_from_training_graph` | no held-out edge in train, val or message passing |
| `held_out_host_has_no_training_edges` (`unseen_host`) | the held-out host is isolated in the message-passing graph |
| `message_passing_is_train_edges` / `_within_split_edges` | message passing uses exactly the allowed edges |
| `edge_features_finite`, `edge_feature_names_match` | no NaN or inf; names align with columns |
| `no_identifier_features` | no IP or port column among the features |
| `preprocessor_fit_on_train_edges` | numeric features have mean 0 and std 1 on training edges, so the fit used train only |
| `node_features_carry_no_identity` / `_from_message_passing_edges` | node features are constant, or recomputable from allowed edges only |

**Result: PASSED.** All 17 checks pass on the `random` graph. The 9 `unseen_attack` and 9 `unseen_host` graphs
are each built and validated too (`--no-graph.validate_holdouts` skips them).

Tests: `tests/test_graph.py` (7 tests). They cover alignment, holdouts, transductive mode, degree features, a
deliberately corrupted graph that must fail, and a save/load round trip.

## What the graph looks like
![host graph](../results/graph/random/host_graph.png)

| Property | Value |
|---|---|
| Nodes (hosts) | 776: 51 send flows; 725 only receive |
| Edges (flows) | 190,444: 133,310 train, 28,567 val, 28,567 test, each 77.9% attack |
| Distinct host pairs | 1,026 directed, 1,004 undirected |
| Flows per pair | median 3, mean 186, max 15,772 (192.168.1.30 → .190, all attack); 272 pairs carry a single flow |
| Self-loop flows (host to itself) | 9,865, **all normal** |
| Connected components | 4, with 600, 157, 17 and 2 hosts |
| Distinct neighbours per host | median 1, mean 2.6, max 271 (192.168.1.190); 633 hosts have one neighbour |
| Density (simple graph) | 0.003 |
| Hosts in val / test with no training edge | 59 / 39, touching 69 / 41 edges |

**Shape:** a few hubs and many leaves. Most hosts are endpoints talking to one other host. The traffic is
concentrated on a few hubs:

| Host | Flows | Sent | Neighbours | Attack share | Role |
|---|---|---|---|---|---|
| 192.168.1.30 | 60,569 | 60,551 | 110 | 99.9% | main attacker (scanning, ddos, …) |
| 192.168.1.190 | 50,758 | 3,133 | 271 | 49% | most-attacked victim and hub; also normal traffic |
| 192.168.1.193 | 42,020 | 21,706 | 20 | 91% | attacker (backdoor and others), also a victim |
| 192.168.1.152 | 30,964 | 15,110 | 75 | 45% | mixed: normal hub and attack target |
| 192.168.1.31 / .32 | 30,255 / 22,402 | all | 115 / 25 | 100% | pure attackers |

**The two large components:**
- **The 600-host component** is the attack network: attacker hubs (red) in the centre, connected to many
  victims and normal hosts.
- **The 157-host component** is entirely normal traffic: green, with no attacks at all.

The 17- and 2-host components are small normal islands. All three distributions (neighbours, flows per host,
flows per pair) are heavy-tailed; see `degree_distributions.png`.

## What this means for the GNN stage
1. **The structure gives the label away on the random split.**
   - Only 3 of 1,026 host pairs mix normal and attack flows.
   - Predicting each test flow from its source host's majority training label gives **F1 0.994**
     (`source_host_shortcut_test_f1`), the same leakage docs/01 found with `src_ip`.
   - A GNN only has to recognise which neighbourhood an edge sits in. Random-split scores therefore say almost
     nothing; **`unseen_attack` and `unseen_host` carry the comparison** (docs/03).
2. **Self-loops are a second shortcut.** Every self-loop flow is normal, and an edge classifier that sees both
   endpoint embeddings can spot "same node". The tabular baselines can't, because they never see IPs. Watch
   for it when GNN results look too good. Dropping or flagging self-loops is a one-line ablation.
3. **Heavy multigraph.** Mean aggregation over a host's flows is dominated by its biggest pairs; one pair
   carries 15,772 flows. A GNN should aggregate with mean (or attention), not sum, or neighbour sampling will be
   needed.
4. **Hosts without training edges** (59 in val, 39 in test) have no embedding under `train_only`. Their edges
   are classified from edge features plus the other endpoint, which is exactly the `unseen_host` situation.
5. **Fair comparison:** same splits (verified), same 102 edge features, same holdouts, and the same fixed-FPR
   metrics as the baselines (docs/02 §6).

## Not done yet
- **Heterogeneous graph** (separate node types, e.g. host and service/port, or IoT-device types). Deferred:
  the homogeneous graph comes first.
- **Temporal graph:** impossible, because the file has no timestamps (docs/01).
- **Line graph / flow-as-node graph:** an alternative if edge-centric GNN layers turn out weak.
