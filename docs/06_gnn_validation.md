# 06 · Phase 2: GNN validation and ablation studies

**Status:** complete (results in §7, from `results/phase2/comparison/phase2.md`). The audit's label-invariance
check must be rerun after its fix (§7.1).

**Verdict:** no leakage, but **the GNN gains are not relational learning.** They come from each host's aggregated
traffic profile, which on ToN-IoT acts as host reputation. On the honest protocol (unseen host) Random Forest
remains the best model.

**Question:** Phase 1 (docs/05) gave GraphSAGE/GAT near-perfect scores, including genuine unseen-attack recall of
1.000 at 0.25% FPR, where untuned RF gets 0.849. Do the GNNs learn network relationships, or do they recall which
hosts attack?

Every Phase 2 output goes to `results/phase2/`. `results/gnn`, `results/baselines` and `frozen/baselines_v2` are
not touched.

## 1. Phase 1 warning signs
- **Too good where it should be hardest.** Genuine unseen attacks: GNNs 0.999–1.000 at 0.25% FPR; RF 0.849,
  XGBoost 0.743, MLP 0.628. mitm: 0.99–1.00, against 0.44–0.67.
- **Collapse where reputation is unavailable.** Unseen host, own threshold: GNN recall on 192.168.1.39 is
  0.00–0.13, and SAGE/GAT on .33/.37 reach only 0.37–0.54, while RF stays ≥ 0.99. The GNNs fail exactly when the attacking host
  has no history.
- **Inconsistent operating points.** Unseen host: GAT mean recall at 0.25% FPR is 0.998, but its own-threshold
  recall falls to 0.37 on some hosts. The val-chosen fixed-FPR threshold likely sits extremely low because val
  normal scores are almost all ≈ 0. So the actual test FPR has to be reported next to every fixed-FPR recall.

## 2. Leakage vs shortcut: what the audit separates
**Leakage** is test information reaching training. Leakage is a bug. `scripts/audit_gnn.py` checks every graph:
- **`validate_graph`:** all 17 docs/04 checks, on each of 24 graphs (random, 9 attack types, 9 hosts, 5 pair
  folds).
- **`features_match_baselines`:** edge features are recomputed the baselines' way, independently, and must equal
  `edge_attr`.
- **`test_not_in_message_passing`:** no val or test edge propagates under `train_only`.
- **`node_inputs_ignore_eval_edges`:** perturbing every val and test flow's features and direction leaves all
  node inputs and the convolution graph unchanged.
- **`held_out_host_has_no_inputs` / `held_out_pair_not_connected`:** the held-out host or pair is absent from
  training.
- **`training_ignores_test_labels`:** shuffling the test labels (and perturbing the test features) changes no
  train or val prediction, and changes no prediction at all when only the labels are shuffled. The check runs
  single-threaded and requires exact equality: multithreaded CPU training isn't bit-reproducible (identical
  reruns differ by ~5e-3), which made the first audit run fail on noise (max diff 2.8e-4).

**A shortcut is not a bug.** It's a property of ToN-IoT: training labels legitimately tell the model which hosts
attack, and the test flows come from the same hosts. The audit measures that exposure. The table below was
computed with the audit's `exposure()`, using training labels only:

| Graph | test pairs seen in train | target src = known attacker | target dst = known victim | target pair seen | **host_prior** recall @0.25% (actual test FPR) |
|---|---|---|---|---|---|
| random | 0.998 | 1.000 | 0.999 | 0.999 | **0.807** (0.1%) |
| unseen_attack: mitm | 0.995 | 0.987 | 0.622 | 0.327 | **0.878** (0.1%) |
| unseen_attack: injection | 0.985 | 1.000 | 1.000 | 0.876 | **0.594** (0.1%) |
| unseen_attack: backdoor | 0.997 | 1.000 | 0.999 | 0.999 | **1.000** (0.1%) |
| unseen_host: 192.168.1.39 | 0.994 | 0.000 | 1.000 | 0.000 | 0.361 (0.1%) |
| unseen_host: 192.168.1.30 | 0.681 | 0.000 | 0.998 | 0.000 | 0.294 (0.0%) |
| unseen_pair: fold 0 | 0.657 | 1.000 | 0.999 | 0.000 | 0.524 (0.1%) |
| unseen_pair: fold 2 | 0.808 | 1.000 | 0.996 | 0.000 | 0.406 (0.0%) |

`host_prior` scores a flow only by its endpoints' attack history in the training labels. It uses no flow
features and no learning, yet on mitm it beats untuned RF (0.878 vs 0.669 at 0.25% FPR).
- **"Unseen attack" is not unseen for a graph model.** 95–100% of each held-out type's test flows come from a host
  that commits *other* attacks in training. A GNN only needs to recognise the attacker.
- **Unseen host keeps a destination-side shortcut.** A held-out attacker has no history, but its victims are
  known victims (100% for .39). That signal is weaker, which matches where the GNNs collapse.
- **Almost every test connection is familiar** (99.8% of test flows use a host pair seen in training), so the
  random split can't separate memorising connections from learning behaviour. Hence the new protocol below.

## 3. Experiments
| # | Experiment | What changes | What it tells us |
|---|---|---|---|
| A | **host_prior** (`--gnn.models '["host_prior",...]'`) | not a neural net: endpoint reputation from training labels | the score reachable by reputation alone. A GNN that only matches it has learned nothing more |
| B | **flow features only** = `mlp` (Phase 1) | no graph | the per-flow signal |
| C | **shuffled_structure** | convolution graph randomly rewired, degrees roughly kept; node inputs intact | whether *who connects to whom* matters. No drop means the convolutions contribute nothing |
| D | **topology_only** | no flow features anywhere: degree-only node inputs; the head sees `[h_u, h_v]` only | what position in the graph alone predicts |
| E | **no_propagation** | node inputs (each host's mean traffic) but **zero** conv layers (GCN = SAGE = GAT, so one runs) | host-level aggregate features vs message passing. If E ≈ GNN, the gain is host reputation, not relationships |
| F | **no_self_loops** | self-loop flows removed from train, val and message passing; test unchanged | the self-loop shortcut. `self_loop_fpr*` and `non_self_loop_fpr*` are reported for **every** run |
| G | **unseen_pair** protocol (new, GNNs **and** RF/XGBoost) | each of 5 folds of undirected host pairs is removed from train, val and message passing; both hosts keep their other connections | generalisation to new connections between known hosts |

**Fixed-FPR verification:** the headline now shows, next to each recall at 0.25% / 1% FPR:
- the **actual test FPR** at that val-chosen threshold;
- for unseen host and unseen pair, the FPR on the **held-out normal flows** (`holdout_fpr_at_*`).

Every run also records self-loop vs other FPR.

**Model comparison:** host_prior, MLP, GCN, GraphSAGE, GAT, RF and XGBoost in one headline
(`scripts/compare_models.py`):
- RF and XGBoost come from `frozen/baselines_v2`, untuned;
- for unseen_pair, RF and XGBoost are run fresh with the same code into `results/phase2/baselines`.

No tuning, and no RL.

## 4. How to read the results
| Outcome | Interpretation |
|---|---|
| GNN ≈ host_prior and ≈ no_propagation, and shuffled_structure barely hurts | the gain is **host reputation**, not relational learning. GNNs can't be claimed to beat RF on unseen attacks |
| shuffled_structure hurts clearly, and GNN > no_propagation | message passing over real neighbours adds information |
| topology_only scores high | structure position alone is predictive, a dataset artefact on random/attack protocols |
| GNN ≫ RF on unseen_pair **and** unseen_host | genuine generalisation beyond familiar hosts and connections (the strongest claim available) |
| no_self_loops changes the random-split FPR | the self-loop shortcut was being used |

The scientifically defensible comparison is **unseen_host** and **unseen_pair** at the same actual FPR. Random
split and unseen_attack mostly measure reputation on this dataset.

## 5. Code
- **`cyberdefense.data.protocols`:** `unseen_pair` and `pair_fold`. A stable MD5 hash of the undirected pair
  makes the folds identical for GNNs and baselines.
- **`cyberdefense.gnn.shortcut.host_prior_proba`:** the reputation reference model.
- **`cyberdefense.gnn.models`:** `rewire`, `use_edge_features`, and `num_layers=0`.
- **`cyberdefense.gnn.train`:** `prepare_inputs(ablation=...)` and `without_self_loops`.
- **`cyberdefense.gnn.experiment`:** routes `--gnn.ablation` (results in `gnn_<ablation>`, only the affected
  models run) and computes `self_loop_fprs`.
- **`cyberdefense.gnn.audit`** and `scripts/audit_gnn.py`: the leakage audit and exposure numbers.
- **`metrics.fixed_fpr_metrics`:** adds `holdout_fpr_at_*`.
- **`reporting.headline_table`:** adds the actual-FPR columns and unseen pair.
- **Tests:** `tests/test_phase2.py` (15). It includes a deliberately leaky graph that the audit must catch.

## 6. Run order
```bash
R=results/phase2
# 1. Leakage audit (CPU, a few minutes; exits 1 on any leak)
uv run python scripts/audit_gnn.py --paths.results_dir $R

# 2. Seed-42 diagnostics (MPS). Reference models and all four protocols, including unseen_pair:
uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps --gnn.seeds '[42]' \
  --gnn.models '["host_prior","mlp","gcn","sage","gat"]' \
  --gnn.protocols '["random","unseen_attack","unseen_host","unseen_pair"]'
#    Ablations (each writes $R/gnn_<ablation>/)
for A in shuffled_structure topology_only no_propagation no_self_loops; do
  uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps --gnn.seeds '[42]' \
    --gnn.ablation $A --gnn.models '["mlp","gcn","sage","gat"]' \
    --gnn.protocols '["random","unseen_attack","unseen_host","unseen_pair"]'
done
#    Baselines on the new unseen_pair protocol (5 seeds; RF/XGBoost are quick)
uv run python scripts/train_baselines.py --paths.results_dir $R --baseline.protocols '["unseen_pair"]' \
  --baseline.tasks '["binary"]' --baseline.seeds '[42,43,44,45,46]'

# 3. Five seeds for the experiments that decide the question (seed 42 is reused)
uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps --gnn.resume \
  --gnn.models '["host_prior","mlp","gcn","sage","gat"]' \
  --gnn.protocols '["random","unseen_attack","unseen_host","unseen_pair"]'
for A in shuffled_structure no_propagation; do
  uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps --gnn.resume \
    --gnn.ablation $A --gnn.models '["gcn","sage","gat"]' \
    --gnn.protocols '["random","unseen_attack","unseen_host","unseen_pair"]'
done

# 4. One comparison table across everything
uv run python scripts/compare_models.py --paths.results_dir $R --gnn.report_name phase2 --gnn.compare_with \
  '["frozen/baselines_v2/results/baselines/summary.csv","results/phase2/baselines/summary.csv",
    "results/phase2/gnn/summary.csv","results/phase2/gnn_shuffled_structure/summary.csv",
    "results/phase2/gnn_topology_only/summary.csv","results/phase2/gnn_no_propagation/summary.csv",
    "results/phase2/gnn_no_self_loops/summary.csv"]'
#    -> results/phase2/comparison/phase2.md
```

## 7. Results
5 seeds (42–46) unless marked *1 seed* (topology_only and no_self_loops were seed-42 diagnostics). Test split.
"@0.25%" means the threshold was chosen on val for a 0.25% FPR, shown with the actual test FPR it reached.

### 7.1 Leakage audit: no leakage found
- **All 24 graphs pass every hard check:** alignment, labels, masks, holdouts absent from train, val and message
  passing, features identical to the baselines', node inputs unaffected by val/test flows, held-out host/pair
  disconnected.
- **Test labels:** shuffling them changes predictions by at most 9e-7 (float noise).
- **Why the first run failed:** the check's second part failed (2.8e-4) because multithreaded CPU training isn't
  bit-reproducible: two runs on identical inputs differ by 4.7e-3. Single-threaded, both an identical rerun and a
  run with perturbed test labels and features give **exactly 0** difference. The check now runs this way; rerun
  `scripts/audit_gnn.py` to record it.

### 7.2 Graph ablations: the real graph structure contributes almost nothing
| Model (genuine unseen attack, 6 types) | mean recall @0.25% | worst @0.25% | actual test FPR |
|---|---|---|---|
| Random Forest | 0.849 | 0.651 | 0.17% |
| host_prior (training labels only, no features) | 0.648 | 0.086 | 0.11% |
| MLP (flow features, no graph) | 0.628 | 0.277 | 0.23% |
| GCN / SAGE / GAT | 0.999 / 1.000 / 0.999 | 0.995 / 1.000 / 0.994 | 0.17–0.18% |
| **GCN / SAGE / GAT, shuffled_structure** | 0.998 / 0.999 / 0.998 | 0.990 / 0.994 / 0.990 | 0.15–0.19% |
| **no_propagation** (host profiles, zero conv layers) | 0.998 | 0.991 | 0.15% |
| topology_only (*1 seed*, no flow features at all) | 0.999 / 0.994 / 0.926 | 0.994 / 0.962 / 0.558 | 0.15% |

- **Random rewiring changes nothing.** Who connects to whom isn't what the GNNs use. The same holds on the random
  split (all 1.000) and on unseen pairs (all 1.000).
- **Removing every message-passing layer changes nothing** (no_propagation ≈ GNN). The gain comes from the node
  input itself: each endpoint's mean traffic over its training flows, i.e. a host profile. Under unseen_attack, a
  held-out type's source host still has its *other* attacks in that profile, so its flows look malicious.
- **Graph position alone is enough** (topology_only, *1 seed*). With no flow features, degree-only node inputs
  reach 0.999 on "unseen" attacks: attacker hosts are identifiable by their position in the graph.
- **The GNNs beat raw label lookup** (host_prior 0.648) because the profile is richer than the attack share, but
  it is still host-level information, not relational.

### 7.3 Unseen host (the honest test): Random Forest wins
| Model | recall, own F1 threshold | held-out host's normal flows flagged (own thr.) | held-out ROC-AUC | recall @0.25% (actual test FPR) |
|---|---|---|---|---|
| **Random Forest** | **0.967** | 0.185 | 0.999 | 0.969 (0.20%) |
| XGBoost | 0.972 | 0.246 | 0.999 | 0.870 (0.13%) |
| MLP | 0.893 | 0.244 | 0.997 | 0.780 (0.24%) |
| GCN / SAGE / GAT | 0.606 / 0.653 / 0.676 | 0.197 / 0.178 / 0.366 | 0.992 / 0.999 / 0.999 | 0.897 / 0.983 / 0.998 (0.58–0.68%) |
| host_prior | 0.873 | 0.787 | 0.965 | 0.502 (0.56%) |

- **At their own threshold the GNNs miss a third of a new attacker's flows**; RF misses 3%. On 192.168.1.39, GNN
  recall is 0.00–0.13 (docs/05).
- **The GNNs' high fixed-FPR recall here is not real.** Their val-chosen threshold sits so low that it reaches
  0.58–0.68% overall test FPR (2–3× the target), and it flags **77–89% of the held-out host's own normal flows**
  (MLP 15%; measured on the 2 held-out hosts that have normal test flows, .193 and .30).
- **Ranking quality is a tie** (ROC-AUC 0.999 for SAGE, GAT and RF). The graph adds no separability for a new
  device.

### 7.4 Unseen host pair: all models detect; GNNs raise fewer alarms on new connections at their own threshold
| Model | recall (own thr.) | held-out pair normal FPR (own thr.) | overall test FPR (own thr.) | recall @0.25% (actual test FPR / pair normal FPR) |
|---|---|---|---|---|
| Random Forest | 0.995 | 7.6% | 2.04% | 0.993 (1.78% / 7.1%) |
| MLP | 0.993 | 9.8% | 3.45% | 0.943 (1.25% / 4.6%) |
| **GCN** | 0.999 | **0.1%** | **0.06%** | 1.000 (1.70% / 7.5%) |
| SAGE / GAT | 0.999 / 1.000 | 1.4% / 2.1% | 0.71% / 0.68% | 1.000 (2.4–2.5% / 7–15%) |
| GCN shuffled / no_propagation | 1.000 / 0.999 | 6.4% / 11.7% | 1.16% / 2.53% | 1.000 (2.8% / 15%; 3.4% / 19%) |

- **Every model detects the attacks on held-out pairs** (recall ≥ 0.99), because their source hosts are known
  attackers (exposure: 100%).
- **The only structure signal in Phase 2:** at their own F1 thresholds, the GNNs flag far fewer *normal* flows on
  new connections (GCN 0.1% vs RF 7.6%). Shuffling the graph or removing propagation loses most of that (6–12%).
- **The signal disappears at fixed FPR:** with val-chosen thresholds every model overshoots the 0.25% target
  (actual 1.3–2.5% for the learned models), and the GNNs' pair-normal FPR is no better than RF's.
- **Treat it as a hypothesis, not a result.** It is one model family's threshold behaviour on 5 folds, with
  known hosts on both ends.

### 7.5 Self-loops: not a shortcut
- **Trained with self-loops, every model handles them:** test self-loop FPR is about 0 for the GNNs **and** the
  graph-free MLP (0.4%). Their flow features alone mark them as normal, so the GNNs gain no shortcut from
  `h_u = h_v`.
- **Trained without them** (*1 seed*), models flag 49–100% of the test self-loops (MLP 99.7%, GCN/GAT 49%), which
  raises random-split FPR to 11–24%. That is a distribution-shift effect (an unseen traffic type), not evidence
  of shortcut use.

### 7.6 Fixed-FPR verification
| Protocol (baselines, host_prior, MLP, GCN/SAGE/GAT) | actual test FPR at a 0.25% target | at a 1% target |
|---|---|---|
| random | 0.10–0.25% for every model ✔ | 0.11–1.44% ✔ |
| unseen attack | 0.11–0.23% ✔ | 0.14–1.44% ✔ |
| unseen host | RF 0.20%, XGB 0.13%, **GNNs 0.58–0.68%** ✘ | RF 1.47%, GNNs 1.20–1.25% |
| unseen pair | **0.80–2.5% for every model** ✘ | 1.2–4.8% ✘ |

- Ablation rows aren't in the table; they follow the same pattern, except no_self_loops (11–25%, §7.5).
- **Val-chosen thresholds transfer on random and unseen_attack.**
- **They don't transfer when hosts or pairs are held out:** val normal traffic doesn't represent the new
  traffic. Fixed-FPR recall on unseen host and unseen pair must always be read next to its actual FPR. ROC-AUC
  and own-threshold results are more reliable there.

## 8. Conclusions and recommendation
1. **Is there leakage?** No. The isolation, alignment, mask and holdout checks pass on all 24 graphs, and test
   labels have no influence.
2. **Are the GNN improvements genuine?** Not as relational learning. Shuffled structure ≈ no message passing ≈
   full GNN on random, unseen attack and unseen pair. The gain comes from per-host traffic profiles, which on
   ToN-IoT encode host reputation (attackers are dedicated machines, 99.8% of test connections are familiar).
   The "unseen attack" 1.000 measures "known attacker launches a new attack", not new-attack detection.
3. **On a genuinely new device (unseen host), the GNNs are worse than Random Forest:** own-threshold recall
   0.61–0.68 vs 0.967, the same ROC-AUC, and their fixed-FPR recall comes at 77–89% false alarms on that host.
4. **The only structural signal:** fewer false alarms on new connections between known hosts (§7.4). It is weak
   evidence that needs confirming.

**Recommended direction:**
1. **Report it honestly.** On ToN-IoT, Random Forest is the strongest detector for unseen devices, and GNN gains
   are a host-profile shortcut. Phase 1 numbers stay in docs/05, marked "reputation-confounded".
2. **Make host-disjoint protocols the benchmark:**
   - an unseen attack type **from unseen hosts** (hold out the type and every host that sends it);
   - unseen host;
   - unseen pair, at matched *actual* FPR.

   These are the only ones a graph model can't pass by recognising hosts.
3. **If GNNs continue on ToN-IoT, remove the profile shortcut:**
   - degree-free, inductive node inputs (for example, per-host statistics recomputed only from the flows being
     classified, never from training flows);
   - or a flow-centric graph (flows as nodes, linked when they share a host), so neighbours carry behaviour, not
     identity.
4. **Get data where structure can matter.** Cyberwheel (Plan step 1) or a dataset with mixed-role hosts and
   lateral movement. On ToN-IoT, detection is saturated and attackers are trivially identifiable. The project's
   GNN + RL value is more likely in **defense**, where topology determines which hosts to isolate, than in flow
   detection.
5. Keep tuning and RL integration for after (2)–(4).
