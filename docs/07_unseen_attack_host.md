# 07 · Phase 3: unseen attack from unseen hosts

**Status:** complete. 5 seeds; 90 baseline, 225 GNN/MLP/host_prior and 45 no_propagation runs. Results in §6,
from `results/phase3/comparison/phase3.md`.

**Verdict:** **no evidence that GNNs generalise to unseen attacks from unseen hosts.**
- On the clean types they tie Random Forest.
- Their only large win (+0.49 recall at 0.25% FPR) is confined to the destination-reputation types, and a GNN with
  **zero** message-passing layers reproduces it.
- On the novel-endpoint types they are worse and unstable.
- At deployable (val-chosen) thresholds they run at 8–10× RF's false-positive rate.

**Question:** do GNNs detect an attack type they have never seen, launched by hosts they have never seen? Phase 2
(docs/06) showed that "unseen attack" alone is passed by recognising the attacker host. Here the attacker hosts are
held out too.

Every output goes to `results/phase3/`. Phase 1 and Phase 2 results are untouched. No graph redesign, tuning or RL.

## 1. Protocol: `unseen_attack_host`
For attack type T:
- **Held-out hosts:** every host that sends T, in **any** split, including hosts with a single T flow. The host
  set is computed once from all splits and carried in the holdout name (`mitm@192.168.1.31,...`), so the
  baselines' per-split masks and the graph's mask are identical.
- **Removed from train, val and message passing:** every flow those hosts send **or receive**, plus every T flow
  (`holdout_mask`).
- **Evaluated** (`target_mask`):
  - the T test flows (recall);
  - the held-out hosts' own normal test flows (their FPR).

  The held-out hosts' attacks of *other, seen* types are neither target nor normal.
- **The same code for every model:** RF and XGBoost (`baselines.experiment`), and host_prior, MLP, GCN, SAGE and
  GAT (`gnn.experiment`), over 5 seeds (42–46).

## 2. Feasibility: does ToN-IoT support it?
ToN-IoT has only **19 attacker hosts** (9 with ≥ 100 attack flows), and each attack type comes from 2–6 of them.
Holding out a type with all its senders also removes those senders' other attacks. A type runs only if training
keeps ≥ 3 attack types and ≥ 3 attacker hosts, and test keeps ≥ 100 target flows. These thresholds are in
`BaselineConfig.attack_host_*`; infeasible types are skipped, not forced.

| Type | Held-out hosts | Train rows kept | Train attack flows / types / attacker hosts | Test target flows | Feasible |
|---|---|---|---|---|---|
| backdoor | 3 | 79% | 76,956 / 7 / 14 | 2,807 | ✔ |
| ddos | 4 | 52% | 39,965 / 5 / 8 | 2,999 | ✔ |
| dos | 4 | 38% | 38,308 / 6 / 10 | 2,848 | ✔ |
| injection | 3 | 52% | 40,359 / 7 / 11 | 2,994 | ✔ |
| mitm | 6 | 61% | 66,105 / 8 / 12 | **156** | ✔ (wide CI) |
| password | 2 | 52% | 40,359 / 7 / 11 | 2,979 | ✔ |
| ransomware | 5 | 79% | 76,956 / 7 / 14 | 2,211 | ✔ |
| scanning | 4 | 40% | 24,688 / 6 / 10 | 3,000 | ✔ |
| xss | 2 | 88% | 87,834 / 8 / 17 | 2,271 | ✔ |

**Limitations:**
- **The 9 evaluations aren't independent.** ddos, injection and password are removed by holding out the same pair
  of hosts (192.168.1.30 and .31), which leaves nearly identical training sets. Backdoor and ransomware share their
  hosts (.193, .33, .37). In practice there are about **4 independent host groups**.
- **Large removals.** Holding out .30/.31 removes about half of the training set, so some of a drop is plain data
  loss rather than generalisation failure.
- **mitm** has only 156 target test flows. Its 95% Wilson interval is about ±3–8 points, so read it with the CI.

## 3. Leakage and shortcut audit
**Leakage: none found.** All 9 graphs pass every `validate_graph` and `audit_graph` check:
- held-out type absent from train, val and message passing;
- **every** held-out host isolated (zero node input, no convolution edges);
- features identical to the baselines'.

**Source reputation is gone:** 0% of target flows come from a host seen attacking in training. But two
host-level shortcuts remain. They were measured with `exposure()`, using training labels only:

| Type | target dst = known victim | target dst unseen in train | normal test flows with both ends unseen | **host_prior** recall @0.25% (labels only, no features) | Exposure |
|---|---|---|---|---|---|
| ddos | 0.998 | 0.002 | 0.000 | 0.007 | destination reputation |
| injection | 0.999 | 0.001 | 0.000 | 0.339 | destination reputation |
| password | 0.998 | 0.001 | 0.000 | 0.036 | destination reputation |
| dos | 0.664 | 0.197 | 0.100 | 0.061 | destination reputation (partial) |
| xss | 0.701 | 0.299 | 0.000 | 0.321 | destination reputation (partial) |
| **backdoor** | 0.000 | **1.000** | 0.037 | **1.000** | **novel endpoint** |
| **ransomware** | 0.001 | **0.999** | 0.037 | **1.000** | **novel endpoint** |
| mitm | 0.109 | 0.891 | 0.110 | 0.064 | novel endpoint (partial) |
| **scanning** | 0.173 | 0.265 | 0.000 | **0.000** | **cleanest** |

- **Destination reputation:** a new attacker hitting a victim that received attacks in training.
- **Novel endpoint:** backdoor and ransomware target victims that never appear in training, while only 3.7% of
  normal test flows have both ends unseen. `host_prior` reaches **1.000** on them by flagging "both ends unknown"
  alone (it falls back to the 78% training attack rate, which outranks almost every normal flow). A GNN has the
  same cue, because a held-out host's node input is all zeros.
- **scanning** is the only type with neither shortcut (`host_prior` 0.000). It is the cleanest single test of
  generalisation in this dataset; mitm is partly exposed.

## 4. Metrics: matched actual FPR and uncertainty
Phase 2 showed that val-chosen thresholds don't transfer when hosts are held out: each model lands at a different
actual test FPR, so recalls aren't comparable. Phase 3 therefore reports both:
- **Recall at a val-chosen 0.25% / 1% FPR,** with the actual test FPR it reaches. This is the deployable operating
  point.
- **Recall at a matched actual test FPR** (`metrics.matched_fpr_metrics`). Each model's threshold is set on the
  test normal flows, so every model sits at exactly 0.25% / 1% actual FPR. It is read off each model's test ROC
  curve; nothing is fitted. Ties can only lower the achieved FPR (`matched_fpr_at_*`).
- **Uncertainty:**
  - mean ± std over 5 seeds (model variance);
  - a 95% Wilson interval per run on the held-out recall (`holdout_tpr_lo/hi_at_*`), for sampling uncertainty.
    This is important for mitm.

  Read results per type and per exposure group (§3), not only as a mean over 9 non-independent splits.

## 5. Run order
```bash
R=results/phase3
# 1. Feasibility + leakage/shortcut audit (CPU, a few minutes; covers every protocol incl. Phase 3)
uv run python scripts/audit_gnn.py --paths.results_dir $R

# 2. RF / XGBoost, 5 seeds (90 runs)
uv run python scripts/train_baselines.py --paths.results_dir $R --baseline.protocols '["unseen_attack_host"]' \
  --baseline.tasks '["binary"]' --baseline.seeds '[42,43,44,45,46]'

# 3. host_prior, MLP, GCN, SAGE, GAT, 5 seeds (225 runs, MPS)
uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps \
  --gnn.models '["host_prior","mlp","gcn","sage","gat"]' --gnn.protocols '["unseen_attack_host"]' --gnn.tasks '["binary"]'

# 4. Shortcut check: GNN without message passing on the same protocol (5 seeds, 45 runs)
uv run python scripts/train_gnn.py --paths.results_dir $R --gnn.device mps --gnn.ablation no_propagation \
  --gnn.models '["sage"]' --gnn.protocols '["unseen_attack_host"]' --gnn.tasks '["binary"]'

# 5. Comparison -> results/phase3/comparison/phase3.md
uv run python scripts/compare_models.py --paths.results_dir $R --gnn.report_name phase3 --gnn.compare_with \
  '["results/phase3/baselines/summary.csv","results/phase3/gnn/summary.csv","results/phase3/gnn_no_propagation/summary.csv"]'
```

## 6. Results
### 6.1 Audit
**PASSED.** All 33 graphs pass, 9 of them Phase 3. Every held-out host is isolated, and the label-invariance check
is now exact (identical rerun, shuffled test labels and perturbed test features all give a difference of 0.0).
Feasibility and exposure are as in §2–3.

### 6.2 Headline: recall at matched actual test FPR, by exposure group
Each model's threshold is set on test normal flows, so every model is at the same actual FPR (achieved ≤ 0.238% and
≤ 0.9997%). Group mean over its types; mean ± std over 5 seeds.

| Group (types) | RF | XGB | MLP | GCN | SAGE | GAT | SAGE, no message passing | host_prior |
|---|---|---|---|---|---|---|---|---|
| **@0.25% FPR** | | | | | | | | |
| A dst-reputation (ddos, injection, password) | 0.507 ± .014 | 0.521 | 0.420 | **1.000** | **1.000** | **1.000** | **1.000** | 0.128 |
| B partial dst (dos, xss) | 0.852 ± .017 | 0.834 | 0.811 | 0.824 ± .075 | 0.613 ± .157 | 0.936 ± .021 | 0.918 ± .022 | 0.191 |
| C novel-endpoint (backdoor, ransomware) | **0.954** ± .025 | 0.750 | 0.733 | 0.799 ± .114 | 0.531 ± .303 | 0.123 ± .275 | 0.809 ± .158 | 0.000 |
| **D clean (scanning, mitm)** | 0.503 ± .008 | 0.490 | 0.501 | 0.506 | 0.508 | 0.524 ± .011 | 0.504 | 0.032 |
| **@1% FPR** | | | | | | | | |
| A dst-reputation | 0.998 | 0.588 | 0.632 | 1.000 | 1.000 | 1.000 | 1.000 | 0.128 |
| B partial dst | 0.915 | 0.875 | 0.925 | 0.842 | 0.614 ± .157 | 0.955 | 0.918 | 0.191 |
| C novel-endpoint | 0.999 | 1.000 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 |
| **D clean** | 0.544 ± .006 | 0.499 | 0.504 | 0.516 | 0.523 | 0.549 ± .009 | 0.521 | 0.032 |

**Paired difference vs RF (same seed), matched 0.25% FPR:**

| Group | GCN | SAGE | GAT | SAGE, no message passing | MLP |
|---|---|---|---|---|---|
| A dst-reputation | +0.493 | +0.493 | +0.493 | **+0.493** | −0.086 |
| B partial dst | −0.028 | −0.239 | +0.084 | +0.066 | −0.041 |
| C novel-endpoint | −0.155 | −0.423 | −0.831 | −0.145 | −0.221 |
| D clean | +0.004 | +0.006 | +0.022 (range +0.005…+0.034) | +0.002 | −0.002 |

### 6.3 Per type (matched FPR, with 95% Wilson CI)
- **scanning (clean):** every model detects it: RF 0.989 [0.98–0.99], MLP 0.995 and GNNs 1.000 at 0.25%; all
  ≥ 0.98 at 1%. The flow features alone are enough (MLP), so this clean test can't separate the models.
- **mitm (clean, 156 flows):** **every model fails.** At 1% FPR: GAT 0.099 [0.06–0.16], RF 0.088 [0.05–0.14],
  SAGE 0.046, GCN 0.032, XGB 0.017, MLP 0.009. The CIs overlap. SAGE has the best ROC-AUC (0.904 vs RF 0.850),
  but at no usable operating point.
- **injection / password / ddos (dst-reputation, same .30/.31 hosts):** GNNs 1.000 vs RF 0.381 / 0.291 / 0.848 at
  0.25%. At 1%, RF recovers to 0.998–0.999, while XGB (0.31–0.96) and MLP (0.38–0.97) don't. The no-message-passing
  SAGE also scores 1.000.
- **xss / dos (partial):** SAGE and GAT reach 0.999 on xss (RF 0.828). On dos, GAT 0.873 ≈ RF 0.876, GCN 0.784, and
  SAGE 0.226 ± 0.313 (unstable).
- **backdoor / ransomware (novel endpoint):** RF 0.995 / 0.913. GAT 0.199 ± 0.444 and SAGE 0.796 ± 0.445 on backdoor,
  and the GNNs score 0.05–0.62 on ransomware: whole seeds collapse. host_prior scores 0.000 at matched FPR, because
  3.7% of normal flows also have both ends unseen. So the matched-FPR metric neutralises the novel-endpoint shortcut
  that inflated val-chosen numbers (host_prior 1.000 in §3).

### 6.4 Deployable operating points (val-chosen thresholds)
| | RF | XGB | MLP | GCN | SAGE | GAT |
|---|---|---|---|---|---|---|
| actual test FPR at a 0.25% target | **0.53%** | 0.89% | 3.0% | 4.2% | 5.2% | 4.5% |
| held-out hosts' normal flows flagged (0.25% target) | **12%** | 8% | 19% | 69% | 70% | 73% |
| own F1 threshold: overall test FPR | **0.53%** | 2.0% | 4.1% | 1.8% | 2.1% | 1.9% |
| own F1 threshold: held-out hosts' normal flows flagged | **11%** | 21% | 25% | 16% | 19% | 38% |

The headline "mean recall at 0.25% FPR" of 0.93–0.997 for the GNNs (vs RF 0.69) is **bought with 8–10× the false
positives**. Their validation-chosen thresholds don't transfer to unseen hosts, and they flag about 70% of a new
host's legitimate traffic.

### 6.5 Conclusions
1. **No leakage.**
2. **On the cleanest tests the GNNs tie RF.** Scanning is easy for every model; mitm defeats every model, with
   overlapping CIs. GAT's +0.02 on the clean group is within mitm's sampling uncertainty.
3. **The large GNN gain (+0.49 at 0.25% FPR) appears only where the victim is a known victim**, and a GNN with
   zero propagation layers gets exactly the same gain. It comes from the victim's aggregated traffic profile
   (destination-side context), not from relational message passing. host_prior (labels only) doesn't explain it,
   so the profile adds information beyond the label share. But it is host context that a new victim wouldn't
   have.
4. **Where both endpoints are new** (novel endpoint), the GNNs are worse than RF and unstable across seeds.
5. **Operationally, RF is the safest detector for unseen attackers:**
   - it is the best or tied on 3 of 4 groups at matched FPR;
   - it has the lowest real FPR at deployable thresholds;
   - it has the lowest false-alarm rate on new hosts.
6. **Answer to the Phase 3 question:** on ToN-IoT, **GNNs do not demonstrably generalise to unseen attacks from
   unseen hosts.** Their apparent gains reduce to destination-host context, and they cost reliability elsewhere.

### 6.6 Dataset limitations
- **About 4 independent host groups,** not 9: group A is one experiment repeated three times (hosts .30/.31). Any
  mean over the 9 types is dominated by it, which is why §6.2 groups the results.
- **Only one clean, feasible unseen-attack test (scanning), and it is trivially detectable from flow features.**
  The other partly clean type (mitm) has 156 flows. ToN-IoT can't give a well-powered answer to "new attack from
  new attacker" for graph models.
- **Removing .30/.31 drops about half the training data,** so some failures mix data loss with novelty.
- The matched-FPR operating point uses test normal flows, so it's an evaluation device, not a deployable
  threshold (§6.4 gives the deployable view).

### 6.7 Recommended next direction
1. **Stop the ToN-IoT detection comparison here and report it as is.** RF is the strongest unseen-attacker
   detector, and GNN gains are host-context effects (Phases 2–3).
2. **If GNN detection continues,** test whether destination context is legitimately useful. Evaluate on *new
   victims* as well, holding out the target hosts, and compare against an RF given the same per-host aggregate
   features. If RF plus host features matches the GNN, the graph isn't needed.
3. **Move the graph to where topology matters:** the defense stage (Cyberwheel, Plan step 1), where the policy
   must decide which hosts to isolate, or a dataset with lateral movement and mixed-role hosts.
4. Tuning and RL integration only after (2) or (3).

## 7. How to read them
| Outcome | Interpretation |
|---|---|
| GNNs > RF/XGBoost/MLP at matched FPR on **scanning** (and mitm), with non-overlapping CIs | genuine generalisation to unseen attacks from unseen hosts |
| GNN gains only on backdoor/ransomware, where host_prior ≈ 1.0 | novel-endpoint shortcut, not attack recognition |
| GNN gains only on ddos/injection/password | destination-reputation shortcut |
| GNN ≈ no_propagation | the gain comes from node inputs, not message passing |
| All models collapse on scanning/mitm | no model generalises; the limit of ToN-IoT for this question |
