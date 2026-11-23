# 03 · Final evaluation protocol and baseline freeze

The fixed conditions every model (RF, XGBoost, GNNs, GNN + RL) is evaluated under. They are frozen once the
steps below are done, and later stages must reproduce them exactly.

## Run order
| # | Step | Command | Gate before moving on |
|---|---|---|---|
| 1 | Build and validate the data | `uv run python scripts/prepare_data.py` then `uv run python scripts/validate_data.py` | Validation reports `PASSED` |
| 2 | Re-tune on disjoint holdouts | `uv run python scripts/tune_baselines.py` | `results/tuning_disjoint/*/best_params.json` exist |
| 3 | Final 5-seed runs, untuned | `uv run python scripts/train_baselines.py --baseline.seeds '[42,43,44,45,46]'` | 5 seeds in `results/baselines/summary.csv` |
| 4 | Final 5-seed runs, tuned | `uv run python scripts/train_baselines.py --baseline.use_tuned --baseline.seeds '[42,43,44,45,46]'` | 5 seeds in `results/baselines_tuned_disjoint/summary.csv` |
| 5 | Freeze | `uv run python scripts/freeze_baselines.py --freeze.tag v2` | `frozen/baselines_v2/manifest.json` written; commit it |

**Time estimates** (not measured):
- Step 2: about 15 minutes for Random Forest and 10–25 minutes for XGBoost.
- Step 3: 40 runs per seed, about 4 minutes per seed, so about 20 minutes.
- Step 4: slower, perhaps 2–3×, because the tuned Random Forest uses 394 trees and 35% of features per split.

Each step's run log shows progress.

**Status: done (2026-11-21).**

| Freeze | Contents | Use |
|---|---|---|
| `frozen/baselines_v1/` | 5-seed baselines without the fixed-FPR metrics | superseded; kept for the record |
| **`frozen/baselines_v2/`** | same runs plus recall at 0.25% and 1% FPR (val-chosen) and holdout ROC-AUC | **the reference for every later model** |

- **Results:** analysed in [docs/02](02_baselines.md).
- **First-round results:** in `results/round1/`, outside the freeze.
- **No git commit recorded:** the repository has no commit yet, so the manifest has `commit: null`. Commit,
  then re-freeze with `--freeze.tag v2 --freeze.overwrite` to record one.

**Main protocol-level finding:** the tuned baselines' higher own-threshold recall came from 5–7× more false
positives. At the same validation-chosen FPR, untuned Random Forest is the strongest baseline on every
generalisation test:
- genuine unseen attack types: 0.849 at 0.25% FPR and 0.986 at 1%;
- unseen hosts: 0.969 at 0.25% FPR.

Recall at a fixed FPR is now the headline metric (section 4).

## 1. Data validation
`scripts/validate_data.py` (`cyberdefense.data.validation`) writes `results/validation/report.json`.

**Hard checks** (a failure exits with status 1 and blocks freezing):
- the raw schema is as expected;
- no exact duplicate rows remain after cleaning;
- no row appears in two splits;
- no kept-intact feature group crosses splits;
- class shares match across splits within 1 point.

**Current result: PASSED.**
- Splits are 133,310 / 28,567 / 28,567 rows.
- Row overlap and group overlap are 0.
- The largest class-share deviation is 0.0001.

**What validation found:**
- **30 duplicates were surviving cleaning.** These normal flows differ from another row only in a dropped text
  column. Cleaning now drops the text columns *before* deduplicating, giving 20,599 duplicates removed and
  190,444 rows. A regression test covers this.
- **Provenance:** all attack classes match the counts reported for the official file exactly. Only normal
  traffic differs: 50,000 here vs 300,000 reported. This copy is very likely the official file with the normal
  class subsampled. For a row-level comparison, download the official `Train_Test_Network.csv` from the UNSW
  page and run `--validation.official_csv <path>`. The report then gives the share of this copy's rows found
  in the official file, overall and for attack and normal rows.
- **Host overlap:** 100% of test attack flows come from hosts already seen attacking in training. The random
  split never measures unfamiliar devices; `unseen_host` does.

## 2. Genuine unseen-attack evaluation
The first tuning run optimised mitm/injection/password recall, so their unseen scores were "tuned for". Final
protocol:

| Role | Attack types | Use |
|---|---|---|
| Tuning holdouts | **xss, dos, ddos** | Their validation recall drives hyperparameter search; reported but marked † |
| Genuinely unseen | **mitm, injection, password**, backdoor, ransomware, scanning | Never used in tuning; **these carry the generalisation claim** |

xss, dos and ddos were chosen because they still give a tuning signal (untuned recall 0.80–0.98), unlike
backdoor and scanning (about 1.00). This is now the default (`TuningConfig.holdouts`), writing to
`results/tuning_disjoint`. The first, targeted run is in `results/round1/tuning` for reference.

**Result:**
- **At 0.25% FPR**, untuned Random Forest detects a mean of **0.849** of the 6 genuinely unseen types. On mitm,
  injection and password it detects 0.67–0.75.
- **At 1% FPR**, it detects 0.986 (achieved test FPR 1.35%).
- Untuned XGBoost and both tuned models are lower at both budgets (docs/02 §4).

Safeguards:
- `apply_tuned` reads the holdouts **actually** tuned on from `best_params.json`, so the † flags can't drift
  from what was optimised.
- Tuning refuses to reuse a folder whose saved results used a different objective (holdouts, weights,
  train fraction).
- XGBoost pruning now waits until 2 holdouts are scored (`prune_after_holdouts`). In the first run, the first
  holdout alone pruned 28 of 40 trials.
- The end-of-run report shows `genuine_unseen_recall` and `tuned_for_recall` separately.

## 3. Unseen-host evaluation
**Question:** can the model detect attacks from a device it has never seen?

- **Holdouts:** the 9 attacker hosts with at least 500 attack flows, all splits counted
  (`unseen_host_min_attack_flows`): 192.168.1.30, .31, .32, .193, .33, .37, .39, .36 and .34.
- **Training:** for each host, *every* flow it sends **or receives** is removed from train and val. Attacker
  hosts also appear as destinations (102k attack flows go back to them), so both directions must go.
- **Test:** the full test split.
  - `holdout_recall`: detection of that host's attack flows.
  - FPR on all normal traffic.
  - `holdout_normal_fpr`: the host's own normal flows, where it has any (for example .193).
- **Why it works for this data:** every attack type except backdoor is produced by at least two hosts, so
  holding one out tests a known attack from a new device, not a new attack. Backdoor comes almost entirely from
  .193, so for that host the test also partly measures unseen-attack generalisation. Note this when reading it.
- **Binary task only**, as with `unseen_attack`.

**Result:**
- The 6 large hosts are detected at ≥ 0.98 recall (0.995 on average). Unfamiliar devices barely hurt tabular
  models.
- At 0.25% FPR, untuned Random Forest keeps a mean host recall of 0.969. The tuned models drop to 0.71–0.78
  and score 0 on host .39, because of tied scores (docs/02 §5).
- The weak scores come from hosts with 72–103 test flows (.34, .39), which are noisy.
- For the GNNs, this protocol is mainly a check that they don't do *worse*, which would signal identity
  memorisation (docs/02 §5).

**Leakage protection:** IPs and ports are never model features (docs/01). For tabular models this protocol
tests behavioural generalisation across devices. For the GNNs, where hosts are graph nodes, it is the critical
test that they haven't memorised attacker identity. A held-out host must not appear in the training graph.

## 4. Final metrics, reported as mean ± std over 5 seeds (42–46)
| Protocol | Headline metrics |
|---|---|
| `random`, binary | F1, FPR, MCC (sanity check: saturated); recall at 0.25% and 1% FPR |
| `random`, multiclass | macro-F1, per-class F1 (mitm), MCC |
| `unseen_attack` | **genuine unseen recall at 0.25% and 1% FPR** (mean, worst) and holdout ROC-AUC; own-threshold recall with its FPR; † types separately |
| `unseen_host` | mean and worst host recall at 0.25% and 1% FPR; own-threshold recall with its FPR |

**Same-FPR rule:** thresholds for the fixed FPRs are chosen on *validation* normal traffic only
(`cyberdefense.metrics.fixed_fpr_metrics`), then applied unchanged to test. These are the headline
comparison numbers. Every model, GNNs included, must report them.
- **Report the achieved test FPR next to each recall.** It drifts from the target: 0.04–0.20% at 0.25%, and
  0.6–1.5% at 1%.
- **A model wins only with higher recall at equal or lower achieved FPR.**
- **Configuration:** the targets are `--baseline.fixed_fprs` (default `[0.0025, 0.01]`). The per-run JSON keeps
  the thresholds in `fixed_fpr_thresholds`.

With several seeds, `train_baselines.py` prints `mean ± std` tables, and `summary_aggregated.csv` holds the
same per model, protocol and holdout. Seeds vary model randomness. The split stays fixed (split seed 42), so
every model sees identical data.

## 5. Freeze
`scripts/freeze_baselines.py` (`cyberdefense.freeze`) writes `frozen/baselines_<tag>/`. The `frozen/` folder is
tracked in git, unlike `results/`.

| Content | Purpose |
|---|---|
| `manifest.json` | raw CSV SHA-256, content fingerprint per split, full config, package versions, git commit, checksum of every copied file, seeds per experiment |
| `validation_report.json` | the passed validation |
| `data_metadata.json` | split sizes, class counts, split checks |
| `results/tuning*/<model>/best_params.json`, `trials.csv` | final hyperparameters and their search history |
| `results/baselines*/summary*.csv` | every baseline metric, per run and aggregated |

Freezing refuses to run without a passed validation report and the untuned baseline results. It also won't
overwrite an existing tag unless you pass `--freeze.overwrite`. Commit the working tree first, so the manifest
records a clean commit.

**The GNN stage must call** `cyberdefense.freeze.verify_splits(cfg, Path("frozen/baselines_v2/manifest.json"))`
before evaluating. It raises if any split differs from the frozen one. The GNNs must also use the same
protocols, metrics, seeds and, if tuned, the same tuning holdouts and objective.
