# 02 · Traditional ML baselines (Random Forest, XGBoost)

Reference scores that every GNN and RL model must beat, on exactly the same data, features and metrics.

> **Status: final and frozen as `frozen/baselines_v2/`** (2026-11-21). v2 adds recall at a fixed,
> validation-chosen FPR (0.25% and 1%) to every binary run ([section 4](#4-same-fpr-comparison-headline)).
> All other numbers match v1. Everything comes from the protocol in [docs/03](03_evaluation_protocol.md):
> - the validated 190,444-row split;
> - tuning on xss/dos/ddos only, so mitm/injection/password are genuinely unseen;
> - 5 seeds (42–46), reported as mean ± std.
>
> The first, single-seed round is summarised in the
> [appendix](#appendix-first-round-superseded); its files are in `results/round1/`.

## Summary of findings
Test set, mean over 5 seeds. "Untuned" means default hyperparameters. Details are in [Results](#results).

1. **Attack vs normal on a random split is saturated.** Every model reaches F1 0.997–0.999 with std ≤ 0.0007,
   so this task can't rank models. It is a sanity check only. Untuned Random Forest is best: F1
   **0.9986**, FPR **0.24%**.
2. **Attack-type classification: untuned XGBoost is best**, with macro-F1 **0.957 ± 0.000**.
   - Tuning *lowered* macro-F1 for both models (by 1.2–1.6 points), mainly through mitm.
   - **mitm is still the weak class:** F1 0.78–0.81 untuned, 0.66–0.68 tuned.
   - **xss → ransomware** is still the largest confusion.
3. **Genuinely unseen attacks are the real gap.** Untuned, both models detect a mean of **0.87** of the 6 attack
   types never used in tuning, at 0.2–0.4% FPR. The hard three are much lower:
   - mitm 0.67–0.69;
   - injection 0.74–0.75;
   - password 0.75–0.79.
4. **At the same false-alarm budget, untuned Random Forest is the best baseline everywhere, and tuning made
   things worse.** These are the 5-seed results with thresholds chosen on validation
   ([section 4](#4-same-fpr-comparison-headline)).
   - At **0.25% FPR**, untuned RF detects **0.849** of the genuinely unseen types, against 0.743 for untuned
     XGBoost and 0.54–0.58 for the tuned models. On the hard three (mitm, injection, password) it detects 0.700,
     against 0.15–0.53.
   - At **1% FPR**, untuned RF reaches **0.986**, against 0.90–0.93 for the others. Its achieved test FPR is
     higher (1.35% against 0.6–1.2%), so this gap is partly the larger budget.
   - **Threshold-free ROC-AUC confirms it:** untuned 0.998 vs tuned 0.995–0.997.
   - **Tuned Random Forest's own-threshold 0.962 came from a permissive threshold** (FPR 1.37%), not from
     better separation. The cause is the objective's weak FPR weight (2.0): on validation, the RF search gained
     0.086 objective from recall and paid only 0.018 for the extra FPR.
5. **Unfamiliar devices are not a problem for flow-level models.** With every flow of a host removed from
   training:
   - the 6 large attacker hosts are still detected at **≥ 0.98** recall (0.995 on average);
   - the only weak hosts (.34 and .39) contribute 72 and 97 test flows, so their scores are noisy.
   The models detect attack *behaviour*, not attacker identity. That leaves the GNNs little room to improve on
   unseen hosts; their main opportunity is unseen attack types and mitm.
6. **Seed variance matters.**
   - Some holdouts vary a lot across seeds: unseen password for XGBoost (± 0.14–0.15), unseen injection and
     ddos for tuned RF (± 0.08–0.09) and host .39 for tuned RF (± 0.12).
   - Single-seed comparisons of these holdouts are unreliable, so the GNNs must also be run over 5 seeds.

**Baselines to beat:**
- **Detection, unseen attacks and unseen hosts:** untuned Random Forest, at 0.25% and 1% FPR.
- **Attack-type classification:** untuned XGBoost.

A GNN counts as an improvement only if its recall is higher **at equal or lower achieved FPR**
([section 6](#6-reference-scores-for-the-gnn-stage)).

## Run
```bash
uv run python scripts/prepare_data.py      # once: builds the splits
uv run python scripts/train_baselines.py   # default grid (see below)
uv run tensorboard --logdir results/tensorboard
```
The full final sequence (validate → tune → 5 seeds → freeze) is in [docs/03](03_evaluation_protocol.md).
Useful variants:
```bash
# Imbalance-handling ablation on the random split
uv run python scripts/train_baselines.py --baseline.protocols '["random"]' \
  --baseline.imbalance '["none","class_weight","undersample","oversample","smote"]'
```
Settings: `BaselineConfig`, `RandomForestConfig` and `XGBoostConfig` in `src/cyberdefense/config.py`.

## Method
### Pipeline (scikit-learn)
For every run, `cyberdefense.baselines.models.fit_baseline`:
1. Fits the preprocessor (`log1p` + standardise numeric, one-hot categorical) on that run's **training rows only**.
2. Applies the imbalance strategy to the **training rows only**.
3. Fits the model. XGBoost early-stops on validation log-loss (up to 1,000 rounds, patience 50).
4. Wraps the result as `Pipeline([("preprocess", ...), ("model", ...)])`, so prediction always goes through
   the same transform.
5. For the binary task, tunes the decision threshold on validation to maximise F1, using
   `TunedThresholdClassifierCV(FrozenEstimator(pipeline))`. The model itself is not refitted.

Test data is used only for the final metrics.

### Imbalance handling
The binary task is 78% attack, and multiclass has mitm at 0.5%.

| Strategy | What it does | Use |
|---|---|---|
| `class_weight` (**default**) | Balanced sample weights in the loss, raised to `class_weight_power` | Main setting. The GNNs use class-weighted loss too, so the comparison stays fair. |
| `none` | Plain fit | Shows how much the handling helps. |
| `undersample` | Random undersampling to the smallest class | Ablation. For multiclass this keeps only about 7k training rows. |
| `oversample` | Random oversampling to the largest class | Ablation. |
| `smote` | Synthetic minority rows | Tabular ablation only. Synthetic flows have no hosts, so they can't enter a graph. |

Two more measures apply to every run: **threshold tuning** (binary) and **imbalance-robust metrics**.

### Metrics (`cyberdefense.metrics`)
Accuracy is recorded but never used to compare models. Under 78% attack, always predicting "attack" scores
78% accuracy.

| Task | Headline | Also reported |
|---|---|---|
| Binary (1 = attack) | F1, FPR, MCC | precision, recall (detection rate), FNR, balanced accuracy, ROC-AUC, PR-AUC |
| Multiclass | Macro-F1, MCC | macro precision and recall, weighted F1, balanced accuracy, macro one-vs-rest ROC-AUC, per-class F1, confusion matrix |
| Unseen attack / host | `holdout_recall`: share of the held-out type's (or host's) test attacks still flagged | FPR on all normal traffic, `seen_attack_recall`, `holdout_normal_fpr` (host), all binary metrics |
| Same-FPR (binary, all protocols) | `recall_at_fpr0p25`, `recall_at_fpr1`; for holdouts, `holdout_recall_at_fpr0p25` / `_fpr1` | achieved `fpr_at_fpr*` on val and test, `holdout_roc_auc` (held-out attacks vs all normal test flows) |

Every run reports train, val and test metrics, so overfitting is visible.

**Recall at a fixed FPR** (`cyberdefense.metrics.fixed_fpr_metrics`, `BaselineConfig.fixed_fprs`, default
0.25% and 1%) compares models at the same false-alarm budget:
- **Threshold on validation:** for each target FPR, the threshold is set so that at most that share of
  *validation* normal flows score above it. It is then applied unchanged to test.
- **Achieved FPR is reported:** the test FPR (`test_fpr_at_*`) can differ slightly from the target, so it is
  recorded next to the recall.
- **Why it's needed:** each model's F1-tuned threshold picks its own operating point, so a model can look better
  just by flagging more. At a fixed FPR that's impossible.
- **Same code for the GNNs:** they use the same function and the same validation-only rule.

### Protocols
- **`random`:** the stratified 70/15/15 split (docs/01). Binary and multiclass.
- **`unseen_attack`:** for each of the 9 attack types, that type is removed from train and val. The model is then
  tested on the full test split. Binary only, because a classifier can't name a class it never saw.
- **`unseen_host`:** for each of the 9 attacker hosts with at least 500 attack flows, every flow it sends or
  receives is removed from train and val. The model is then tested on that host's test attack flows. Binary
  only.

The grid is 2 models × (2 tasks on `random` + 9 types + 9 hosts) = **40 runs per seed**, so 200 per
configuration.

### Outputs (`results/baselines/`, `results/baselines_tuned_disjoint/`)
| File | Contents |
|---|---|
| `<protocol>/<task>/<run>.json` | config, row counts, threshold, timings, train/val/test metrics, test confusion matrix, per-class report |
| `<protocol>/<task>/<run>__confusion.png` | row-normalised test confusion matrix (random protocol) |
| `summary.csv` | one row per run, all val and test metrics |
| `summary_aggregated.csv` | mean and std over seeds per model/task/protocol/holdout |
| TensorBoard `baselines*/` | XGBoost per-round train/val curves, metrics, confusion matrix, HPARAMS table |

## Hyperparameter tuning
**Goal:** improve the baselines on attack types they haven't seen while keeping false positives low, so the GNNs
face a strong reference. The tuning is deliberately controlled: fixed budgets and validation data only.

### Run
```bash
uv run python scripts/tune_baselines.py                         # holdouts xss/dos/ddos -> results/tuning_disjoint
uv run python scripts/train_baselines.py --baseline.use_tuned   # -> results/baselines_tuned_disjoint
```
Variants:
```bash
uv run python scripts/tune_baselines.py --tuning.models '["xgboost"]' --tuning.xgb_trials 60
uv run python scripts/tune_baselines.py --tuning.train_fraction 0.5    # faster trials
uv run python scripts/tune_baselines.py --tuning.timeout_minutes 30    # wall-clock cap per model
uv run python scripts/tune_baselines.py --tuning.weights.fpr 10        # punish false positives harder
```
Rerunning the XGBoost search resumes `study.db` up to `xgb_trials`. Delete the file to start fresh.

### Objective (validation split only)
Each trial fits:
1. one **multiclass** model on train → **val macro-F1**, plus per-class F1 for the holdouts;
2. one **binary** model per tuning holdout (**xss, dos, ddos**), trained and threshold-tuned **without** that
   attack type → **val recall on it** and **val FPR** on normal traffic.

```
objective = 0.5 · macro_F1 + 0.5 · mean unseen recall − 2.0 · mean FPR
```
A 1-point rise in FPR costs 0.02, the same as a 4-point drop in unseen recall. The results show this weight
is **too weak** ([section 4](#4-same-fpr-comparison-headline)). Every part is logged per trial in
`trials.csv`, and trial 0 is always the untuned default.

### Search methods and spaces (`src/cyberdefense/baselines/tuning.py`)
| Model | Method | Budget | Parameters |
|---|---|---|---|
| Random Forest | Random search (`ParameterSampler`) | 30 trials | `n_estimators` 100–600, `max_depth` {None, 10, 20, 30, 50}, `min_samples_split` 2–20, `min_samples_leaf` 1–10, `max_features` {sqrt, log2, 0.2, 0.35, 0.5}, `max_samples` {None, 0.5, 0.7, 0.9}, `criterion` {gini, entropy, log_loss}, `class_weight_power` 0–1 |
| XGBoost | Optuna TPE (multivariate), median pruning once 2 holdouts are scored | 40 trials | `learning_rate` 0.03–0.3 (log), `max_depth` 3–12, `min_child_weight` 1–20 (log), `subsample` 0.5–1, `colsample_bytree` 0.4–1, `gamma` 0–5, `reg_lambda` 0.001–20 (log), `reg_alpha` 0–5, `max_delta_step` 0–10, `class_weight_power` 0–1 |

`class_weight_power` tunes the imbalance handling itself: weights = balanced^power, where 0 means no weighting
and 1 means fully balanced.

### Outputs (`results/tuning_disjoint/<model>/`)
| File | Contents |
|---|---|
| `trials.csv` | every trial: parameters, objective, macro-F1, per-holdout F1, unseen recall and FPR, seconds |
| `best_params.json` | best parameters, their val scores next to the default's, and the tuning config |
| `study.db` | XGBoost Optuna study (resumable; works with `optuna-dashboard`) |

## Results
**Setup:**
- Data: 133,310 train, 28,567 val and 28,567 test rows. The test set has 6,302 normal and 22,265 attack flows.
- Seeds: 42–46, with the split fixed.
- Imbalance handling: `class_weight` throughout, plus a validation-tuned threshold for the binary task.
- Result folders:
  - Untuned: `results/baselines/`
  - Tuning: `results/tuning_disjoint/`
  - Tuned: `results/baselines_tuned_disjoint/`

Values are mean ± std over seeds; a std is omitted where it is below 0.001.

### 1. Attack vs normal, random split (test)
| Model | F1 | Precision | Recall | FPR | MCC | False alarms / missed attacks (per seed) | Recall @ 0.25% FPR (achieved) | Recall @ 1% FPR (achieved) |
|---|---|---|---|---|---|---|---|---|
| **RF untuned** | **0.9986 ± 0.0001** | 0.9993 | 0.9978 | **0.24% ± 0.04** | **0.9935** | 15 / 49 | **0.998** (0.20%) | 0.999 (1.4%) |
| XGB untuned | 0.9983 ± 0.0007 | 0.9982 | 0.9983 | 0.63% ± 0.44 | 0.9921 | 39 / 38 | 0.972 ± 0.032 (0.10%) | 0.999 (0.8%) |
| RF tuned | 0.9971 ± 0.0006 | 0.9963 | 0.9979 | 1.31% ± 0.50 | 0.9867 | 83 / 47 | 0.978 ± 0.008 (0.20%) | 0.995 (0.6%) |
| XGB tuned | 0.9973 ± 0.0001 | 0.9969 | 0.9976 | 1.10% ± 0.03 | 0.9876 | 69 / 52 | 0.948 ± 0.011 (0.04%) | 0.998 (1.2%) |

The task is saturated: PR-AUC is 0.9999–1.0000 for every model. Tuning left missed attacks unchanged and raised
false alarms 2–5×, which is the same shift seen in the unseen-attack results.

### 2. Attack type, random split (test)
| Model | Macro-F1 | Balanced acc. | MCC | ROC-AUC (macro) | Weighted F1 | Train macro-F1 |
|---|---|---|---|---|---|---|
| RF untuned | 0.9540 ± 0.0004 | 0.9572 | 0.9699 | 0.9963 | 0.9736 | 0.9862 |
| RF tuned | 0.9378 ± 0.0002 | 0.9548 | 0.9668 | 0.9977 | 0.9716 | 0.9396 |
| **XGB untuned** | **0.9566 ± 0.0003** | **0.9704** | **0.9756** | **0.9994** | **0.9789** | 0.9691 |
| XGB tuned | 0.9438 ± 0.0004 | 0.9673 | 0.9713 | 0.9991 | 0.9755 | 0.9469 |

Per-class F1 (test, mean of 5 seeds):

| Class | RF untuned | RF tuned | XGB untuned | XGB tuned |
|---|---|---|---|---|
| **mitm** (156 test rows) | **0.808** | 0.655 | 0.781 | 0.680 |
| xss | 0.917 | 0.912 | **0.929** | 0.920 |
| ransomware | 0.940 | 0.940 | 0.941 | 0.939 |
| injection | 0.960 | 0.962 | **0.976** | 0.973 |
| ddos | 0.969 | 0.972 | **0.984** | 0.979 |
| dos | 0.982 | 0.976 | **0.984** | 0.980 |
| scanning | 0.986 | 0.985 | **0.988** | 0.987 |
| password | 0.987 | 0.989 | **0.991** | 0.990 |
| normal | 0.991 | 0.988 | **0.993** | 0.990 |
| backdoor | 0.999 | 0.998 | 0.999 | 0.999 |

Largest confusions (true → predicted, test counts summed over 5 seeds), the same in every model:
- xss → ransomware: about 660 (about 132 per seed)
- injection → xss: 300–510
- xss → injection and ransomware → xss: about 250 each
- normal → ransomware / ddos: 240–320

**Reading:**
- **Tuning lowered macro-F1** for both models. The searches optimised unseen-attack recall, and the multiclass
  model paid for it, mostly on mitm (RF 0.808 → 0.655).
- **Untuned RF overfits** (train 0.986 vs test 0.954). Tuning removed that gap (0.940 vs 0.938), but by
  underfitting: `max_depth` 20 and `min_samples_leaf` 9.
- **The xss/ransomware/injection confusion** matches the conflicting-label groups found in docs/01: identical
  flow features with different labels. No flow-only model can separate them.

### 3. Unseen attacks (test)
Recall on each held-out type when the model was trained without it:

| Held-out type | Test flows | RF untuned | RF tuned | XGB untuned | XGB tuned |
|---|---|---|---|---|---|
| **mitm** | 156 | 0.665 ± 0.031 | **0.882 ± 0.009** | 0.692 ± 0.039 | 0.701 ± 0.042 |
| **injection** | 2,994 | 0.751 ± 0.004 | **0.894 ± 0.082** | 0.738 ± 0.006 | 0.749 ± 0.016 |
| **password** | 2,979 | 0.789 ± 0.066 | **0.998** | 0.753 ± 0.141 | 0.886 ± 0.151 |
| backdoor | 2,807 | 0.999 | 0.999 | 0.999 | 0.999 |
| ransomware | 2,211 | 1.000 | 1.000 | 1.000 | 1.000 |
| scanning | 3,000 | 0.999 | 0.999 | 0.999 | 0.998 |
| xss † | 2,271 | 0.711 ± 0.042 | 0.997 | 0.850 ± 0.049 | 0.925 ± 0.043 |
| dos † | 2,848 | 0.931 ± 0.002 | 0.900 ± 0.002 | 0.946 ± 0.002 | 0.910 ± 0.011 |
| ddos † | 2,999 | 0.721 ± 0.004 | 0.884 ± 0.094 | 0.984 ± 0.008 | 0.991 ± 0.005 |
| **Genuine unseen, mean of 6** | | 0.867 ± 0.006 | **0.962 ± 0.014** | 0.864 ± 0.027 | 0.889 ± 0.030 |
| Hard three (mitm, injection, password) | | 0.735 ± 0.012 | **0.925 ± 0.029** | 0.728 ± 0.053 | 0.779 ± 0.060 |
| Worst genuine unseen (mitm) | | 0.665 | **0.882** | 0.692 | 0.701 |
| Tuned-for †, mean of 3 | | 0.788 | 0.927 | 0.927 | 0.942 |
| **Mean FPR** | | **0.21%** | 1.37% | 0.36% | 1.13% |
| Worst FPR | | **0.29%** | 1.75% | 0.59% | 1.43% |

† marks a tuning holdout: its validation rows chose the tuned hyperparameters, so the *tuned* columns are
optimistic for it. Attacks the model has seen are still detected 99.6–99.8% of the time in every run.

**Reading:**
- **Untuned, the two models are equal on genuinely unseen types** (0.867 vs 0.864). Both miss about a third
  of mitm and a quarter of injection and password.
- **Tuned RF's numbers look excellent but come with 6.5× more false alarms.** At the same FPR the gain
  disappears (section 4).
- **XGBoost's unseen password recall is unstable**: seeds range from about 0.55 to 1.00 (± 0.14–0.15).
- **mitm has only 156 test flows**, so a single flow moves its recall by 0.6 points.

### 4. Same-FPR comparison (headline)
**Method:**
- For each run, the threshold is set on the **validation** normal flows so that at most 0.25% (or 1%) of them
  score above it. It is then applied unchanged to test (`cyberdefense.metrics.fixed_fpr_metrics`).
- Each value is the mean over 5 seeds, with the achieved test FPR in brackets.
- This replaces the earlier seed-42 diagnostic, whose threshold was set on test normal flows.

**Genuinely unseen attack types** (never used in tuning):

| | RF untuned | XGB untuned | RF tuned | XGB tuned |
|---|---|---|---|---|
| **Mean of 6 @ 0.25% FPR** | **0.849 ± 0.009** (0.18%) | 0.743 ± 0.040 (0.10%) | 0.581 ± 0.034 (0.16%) | 0.539 ± 0.072 (0.04%) |
| Hard three @ 0.25% FPR | **0.700 ± 0.018** | 0.490 ± 0.080 | 0.530 ± 0.051 | 0.153 ± 0.038 |
| Worst type @ 0.25% FPR | **0.669** (mitm) | 0.404 (injection) | 0.234 (scanning) | 0.082 (injection) |
| **Mean of 6 @ 1% FPR** | **0.986 ± 0.003** (1.35%) | 0.911 ± 0.016 (0.60%) | 0.932 ± 0.004 (0.65%) | 0.898 ± 0.021 (1.17%) |
| Hard three @ 1% FPR | **0.972 ± 0.006** | 0.823 ± 0.031 | 0.873 ± 0.003 | 0.796 ± 0.042 |
| Worst type @ 1% FPR | **0.928** (mitm) | 0.757 (injection) | 0.753 (injection) | 0.742 (injection) |
| ROC-AUC, holdout vs normal (mean of 6) | **0.998** | **0.998** | 0.997 | 0.995 |
| *Own threshold, for reference* | 0.867 (0.21%) | 0.864 (0.36%) | 0.962 (1.37%) | 0.889 (1.13%) |

Per held-out type, recall @ 0.25% / @ 1% FPR:

| Held-out type | RF untuned | XGB untuned | RF tuned | XGB tuned |
|---|---|---|---|---|
| **mitm** | **0.669** / **0.928** | 0.565 / 0.844 | 0.414 / 0.872 | 0.126 / 0.769 |
| **injection** | **0.748** / **0.999** | 0.404 / 0.757 | 0.695 / 0.753 | 0.082 / 0.742 |
| **password** | **0.682** / 0.990 | 0.499 / 0.869 | 0.481 / **0.994** | 0.252 / 0.878 |
| backdoor | 0.999 / 0.999 | 0.994 / 0.999 | 0.998 / 0.999 | 0.992 / 0.999 |
| ransomware | 1.000 / 1.000 | 1.000 / 1.000 | 0.661 / 0.973 | 1.000 / 1.000 |
| scanning | 0.999 / 1.000 | 0.995 / 0.999 | 0.234 / 0.999 | 0.785 / 0.998 |
| xss † | 0.707 / 0.983 | 0.569 / 0.930 | 0.588 / 0.992 | 0.523 / 0.957 |
| dos † | 0.932 / 0.956 | 0.934 / 0.951 | 0.880 / 0.901 | 0.771 / 0.927 |
| ddos † | 0.722 / 0.998 | 0.618 / 0.998 | 0.708 / 0.754 | 0.526 / 0.996 |

**Reading:**
- **Untuned RF wins clearly at the strict budget.** At 0.25% FPR it leads every genuinely unseen type. Its
  achieved FPR (0.18%) is still below the target.
- **At 1%, untuned RF is ahead but the budgets aren't equal.** Its threshold lets through more false alarms
  on test (1.35%) than the others' (0.6–1.2%), because validation and test normal traffic differ in the tail.
  That's why the achieved FPR must always be reported. Even so, the lead holds on the hard three (0.97 vs ≤ 0.87).
- **Tuning bought no separation.** Untuned models have equal or higher ROC-AUC (0.998 vs 0.995–0.997). The
  tuned models' own-threshold recall came from their permissive F1 threshold.
- **The tuned models collapse at 0.25% for a second reason: tied scores.**
  - Shallower trees and bigger leaves (tuned RF: `min_samples_leaf` 9; tuned XGBoost: depth 5) give many flows
    exactly the same score.
  - A flow counts as an attack only if it scores *above* the threshold. So a block of attacks tied with the
    threshold is entirely missed; for example, tuned RF on scanning drops to 0.234.
  - Coarse scores are a real weakness at strict budgets, so this counts against the tuned models. It also means
    their recall at fixed FPR is very sensitive to the target.
- **XGBoost is unstable at 0.25%:** injection 0.404 ± 0.299 and ddos 0.618 ± 0.182 across seeds.

### 5. Unseen hosts (test)
Recall on each held-out attacker host when every flow it sent or received was removed from training:

| Held-out host | Test attack flows | RF untuned | RF tuned | XGB untuned | XGB tuned |
|---|---|---|---|---|---|
| 192.168.1.30 | 9,050 | 0.994 | 0.998 | 0.998 | 0.997 |
| 192.168.1.193 | 5,780 | 1.000 | 1.000 | 1.000 | 1.000 |
| 192.168.1.31 | 4,602 | 0.994 | 0.995 | 0.996 | 0.996 |
| 192.168.1.32 | 3,322 | 0.981 ± 0.004 | 0.998 | 0.985 | 0.988 ± 0.004 |
| 192.168.1.33 | 2,576 | 1.000 | 1.000 | 1.000 | 1.000 |
| 192.168.1.37 | 2,442 | 0.999 | 0.999 | 0.999 | 0.999 |
| 192.168.1.36 | 103 | 1.000 | 1.000 | 0.998 ± 0.004 | 0.996 ± 0.005 |
| 192.168.1.39 | 97 | 0.990 | 0.703 ± 0.121 | 0.977 ± 0.005 | 0.967 ± 0.028 |
| 192.168.1.34 | 72 | 0.747 ± 0.037 | 0.917 | 0.797 ± 0.092 | 0.769 ± 0.061 |
| **Mean, 9 hosts** | | 0.967 ± 0.004 | 0.957 ± 0.013 | **0.972 ± 0.010** | 0.968 ± 0.005 |
| Mean, 6 large hosts | | 0.995 | 0.998 | 0.996 | 0.997 |
| Worst host | | .34: 0.747 | .39: 0.703 | .34: 0.797 | .34: 0.769 |
| **Mean FPR** | | **0.23%** | 1.45% | 0.48% | 1.11% |
| **Mean @ 0.25% FPR** (achieved) | | **0.969 ± 0.004** (0.20%) | 0.782 ± 0.006 (0.18%) | 0.870 ± 0.072 (0.13%) | 0.706 ± 0.008 (0.06%) |
| Mean @ 1% FPR (achieved) | | **0.991 ± 0.002** (1.47%) | 0.951 ± 0.005 (0.69%) | 0.987 ± 0.008 (0.70%) | 0.977 ± 0.002 (1.20%) |
| Worst host @ 0.25% FPR | | **.34: 0.764** | .39: 0.000 | .34: 0.511 | .39: 0.000 |

**Reading:**
- **Unfamiliar devices barely hurt detection.** Every host with thousands of flows stays at ≥ 0.98 recall, the
  same as on the random split. The attacks are recognisable by their behaviour, and with IPs excluded from the
  features there is no identity to memorise.
- **The weak scores come from small hosts:** .34 (72 flows) and .39 (97 flows) move 1–1.4 points per flow and
  swing ± 0.04–0.12 across seeds. Don't build conclusions on them.
- **The held-out host's own normal traffic** is flagged more often than normal traffic overall: about 4% for .193
  vs 0.2–1.5% overall. The 33–44% for .30 rests on only 9 normal flows.
- **At the same FPR, untuned RF stays robust and the tuned models break.**
  - Untuned RF holds 0.969 at 0.25% FPR.
  - Tuned RF and tuned XGBoost both score **0** on host .39: all 97 of its flows tie at or below the threshold.
  - Untuned XGBoost swings across seeds on the small hosts (.39: ± 0.42).
- **For the GNNs:** a GNN has little room to improve on this protocol. Its job here is to show it **does not
  lose** on unseen hosts, which proves it isn't memorising node identity. A score clearly below these baselines
  would signal identity leakage.

### 6. Reference scores for the GNN stage
These are frozen in `frozen/baselines_v2/`. Same-FPR scores are the headline. The achieved test FPR is in
brackets; a GNN must match or beat it.

| Evaluation | Score to beat | Model |
|---|---|---|
| **Genuine unseen, mean of 6 @ 0.25% FPR** | **0.849 ± 0.009** (0.18%) | RF untuned |
| Genuine unseen, hard three @ 0.25% FPR | 0.700 ± 0.018; mitm 0.669, injection 0.748, password 0.682 | RF untuned |
| **Genuine unseen, mean of 6 @ 1% FPR** | **0.986 ± 0.003** (1.35%) | RF untuned |
| Genuine unseen, hard three @ 1% FPR | 0.972 ± 0.006; worst mitm 0.928 | RF untuned |
| Genuine unseen, ROC-AUC (holdout vs normal) | 0.998 (hard three 0.996) | RF and XGB untuned |
| **Unseen hosts, mean @ 0.25% / 1% FPR** | **0.969** (0.20%) / **0.991** (1.47%) | RF untuned |
| Attack vs normal, recall @ 0.25% FPR | 0.998 (0.20%) | RF untuned (saturated) |
| **Attack type, macro-F1** | **0.957 ± 0.000** | XGB untuned |
| mitm F1, multiclass | 0.808 | RF untuned |
| *Own threshold, for reference* | genuine unseen 0.867 at 0.21% FPR; tuned RF 0.962 at 1.37% FPR | |

**Comparison rules:**
1. **Headline:** recall at 0.25% and 1% FPR, with thresholds chosen on validation normal traffic using
   `cyberdefense.metrics.fixed_fpr_metrics`. Report the achieved test FPR next to every recall. A GNN beats a
   reference only with higher recall at **equal or lower achieved FPR**.
2. Mean ± std over seeds 42–46, on the frozen splits (`verify_splits` against `frozen/baselines_v2/manifest.json`).
3. If the GNNs are tuned, they use the same holdouts (xss/dos/ddos), the same objective and validation data only.
4. Also report the threshold-free holdout ROC-AUC. It isn't affected by tied scores or by validation and test
   normal traffic differing.

### 7. Tuning: what the searches found (validation)
| | RF default | RF tuned | XGB default | XGB tuned |
|---|---|---|---|---|
| Objective | 0.864 | **0.927** (trial 21) | 0.930 | **0.935** (trial 19) |
| Macro-F1 | 0.952 | 0.943 | 0.954 | 0.944 |
| Unseen recall, xss / dos / ddos | 0.682 / 0.938 / 0.730 | 0.997 / 0.877 / 0.994 | 0.853 / 0.946 / 0.994 | 0.952 / 0.920 / 0.996 |
| Mean FPR | 0.22% | 1.10% | 0.66% | 0.74% |
| Trials | 30 random (objective 0.857–0.927) | | 40 TPE: 12 complete, 28 pruned | |

**Best parameters** (in `results/tuning_disjoint/<model>/best_params.json`):

| Random Forest | Value (default) |
|---|---|
| `n_estimators` | 458 (300) |
| `max_depth` | 20 (None) |
| `min_samples_split` | 4 (2) |
| `min_samples_leaf` | 9 (1) |
| `max_features` | sqrt (sqrt) |
| `max_samples` | 0.9 (None) |
| `criterion` | gini (gini) |
| `class_weight_power` | **0.11** (1.0) |

| XGBoost | Value (default) |
|---|---|
| `learning_rate` | 0.18 (0.1) |
| `max_depth` | 5 (8) |
| `min_child_weight` | 1.24 (1) |
| `subsample` | 0.94 (0.8) |
| `colsample_bytree` | 0.93 (0.8) |
| `gamma` | 1.76 (0) |
| `reg_lambda` | 0.14 (1) |
| `reg_alpha` | 2.39 (0) |
| `max_delta_step` | 8 (0) |
| `class_weight_power` | **0.60** (1.0) |

**Observations:**
- **Both searches chose simpler models.** RF went shallower with bigger leaves; XGBoost went shallower with
  L1, gamma and `max_delta_step` regularisation. Smoother decision functions score unfamiliar attacks as attack
  more readily, which is what the objective rewarded.
- **Class weighting went down again** (RF 0.11, XGB 0.60), confirming the first round's finding that full
  balancing over-weights the rare classes.
- **Validation predicted test well**: tuned RF's val unseen recall on xss was 0.997, and test gave 0.997. The
  problem is the objective's trade-off, not overfitting to validation.
- **The XGBoost search barely moved** (objective 0.930 → 0.935), and pruning still stopped 28 of 40 trials.
  The untuned XGBoost was already near the best this objective could find.

### 8. Limitations
1. **The objective under-penalises false positives** (section 4). Rerunning tuning with
   `--tuning.weights.fpr 10` or higher, or adding a hard FPR cap, would likely give a tuned model that is better
   at the low-FPR operating point. It is optional; the low-FPR reference is untuned and needs no tuning.
2. **Small test sets:** mitm (156 flows) and hosts .34, .36 and .39 (72–103 flows) are noisy. Quote them with
   their std.
3. **Flow-level ceiling:** the xss/ransomware/injection confusion and mitm precision come from identical flow
   features. Host context (the GNN stage) is the natural next lever.
4. **Dataset copy:** normal traffic is subsampled (50k vs 300k in the official file; docs/03). The FPR estimates
   rest on 6,302 test normal flows. At 0.25%, that budget is only about 16 test false alarms, and the
   validation threshold rests on about 16 validation normals, so recall at 0.25% is noisy.
5. **The achieved FPR drifts from the target:**
   - at the 0.25% target, achieved test FPR ranges from 0.04% to 0.20%;
   - at the 1% target, from 0.6% to 1.5%.
   Validation and test normal traffic differ in the tail, and tied scores push FPR below the target. Always
   compare at the achieved FPR, not the nominal one.
6. **Tied scores:** recall at a fixed FPR counts a flow only if it scores *above* the threshold, so models with
   coarse scores (the tuned ones) lose whole blocks of tied attacks. That's a fair penalty for a deployable
   threshold. An interpolated ROC point (random tie-breaking) would be gentler. It is optional; the ROC-AUC
   already shows the same ranking.

## Appendix: first round (superseded)
- **Setup:** seed 42 only, on the pre-fix 190,474-row split. Tuning targeted mitm/injection/password
  (`results/round1/`).
- **Headline claim:** tuned RF raised mean unseen recall from 0.937 to 0.965 *and* cut FPR from 1.16% to 0.37%.
- **That claim did not hold up in the final protocol:**
  - final untuned RF already runs at 0.21% FPR (std 0.0004 over 5 seeds);
  - tuning now *raises* FPR.
- **Cause unknown:** the first round's untuned per-run files were overwritten by the final run, so the
  difference can't be traced.
- **Findings that replicated:**
  - binary saturation;
  - mitm as the weak class;
  - the xss/ransomware confusion;
  - weak class weighting chosen by tuning.
