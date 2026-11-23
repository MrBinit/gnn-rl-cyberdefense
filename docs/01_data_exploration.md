# 01 · ToN-IoT data: exploration, cleaning and split

## Summary
- **Data:** ToN-IoT network train/test CSV. 211,043 flows → **190,444 after removing exact duplicates**.
  33 flow features, binary `label` and 10-class `attack_type`.
- **Split:** 70/15/15 train/val/test, **stratified on attack type** (identical class shares in all splits)
  and group-aware on identical feature vectors. Saved as parquet in `data/processed/ton_iot/`.
- **Leakage:** the source IP alone predicts the label (F1 0.994). IPs and ports are kept for graph
  construction only, **never as features**.
- **Difficulty:** binary detection on a random split is saturated (F1 ≈ 0.999). The research comparison
  must rely on **unseen attack types**, **unseen attacker hosts** and **multiclass macro-F1**.
- **Graph design:** a homogeneous host graph with flows as edges, and edge classification.

## Reproduce
```bash
uv run python scripts/prepare_data.py                              # clean → split → parquet → figures
uv run python scripts/prepare_data.py --no-ton_iot.make_figures # skip figures
```
| Output | Contents |
|---|---|
| `data/processed/ton_iot/train.parquet`, `val.parquet`, `test.parquet` | the three splits |
| `data/processed/ton_iot/clean.parquet` | all cleaned rows, plus a `split` column |
| `data/processed/ton_iot/metadata.json` | row and class counts per split, split config, checks |
| `results/eda/*.png`, `results/eda/summary.json` | the figures in this document, plus graph statistics |

All settings are in `TonIoTConfig` and `SplitConfig` (`src/cyberdefense/config.py`). Load a split in code
with `cyberdefense.data.ton_iot.load_split(cfg.ton_iot, "train")`.

## Source
`train_test_network.csv` from the Hugging Face copy of the ToN-IoT network train/test set
(`codymlewis/TON_IoT_network`), 211,043 rows × 44 columns. The official source is the UNSW ToN-IoT
project page, in its `Train_Test_datasets` folder.

**Licence:** free for academic research. Commercial use requires permission from the author
(Dr Nour Moustafa). Publications must cite the ToN-IoT papers listed on the dataset page.

**Provenance check** (`scripts/validate_data.py`, results/validation/report.json): every attack class matches
the counts reported for the official file exactly (8 × 20,000 plus 1,043 mitm). Only the normal class differs:
**50,000 here vs 300,000 reported** (461,043 rows in total). This copy is very likely the official file with
normal traffic subsampled. That reverses the class balance: 78% attack here, about 35% in the official file.
A row-level comparison needs the official CSV, which has to be downloaded by hand:
`uv run python scripts/validate_data.py --validation.official_csv <path>`. Until that check is done, state
the subsample in any write-up.

### Raw columns (44)
| Group | Columns |
|---|---|
| Identifiers (4) | `src_ip`, `src_port`, `dst_ip`, `dst_port` |
| Numeric (11) | `duration`, `src_bytes`, `dst_bytes`, `missed_bytes`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`, `http_trans_depth`, `http_request_body_len`, `http_response_body_len` |
| Categorical (22) | `proto`, `service`, `conn_state`, `dns_qclass`, `dns_qtype`, `dns_rcode`, `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`, `ssl_version`, `ssl_cipher`, `ssl_resumed`, `ssl_established`, `http_method`, `http_version`, `http_status_code`, `http_orig_mime_types`, `http_resp_mime_types`, `weird_name`, `weird_addl`, `weird_notice` |
| Free text (5, dropped) | `dns_query`, `http_uri`, `http_user_agent`, `ssl_subject`, `ssl_issuer` |
| Targets (2) | `label` (0 normal / 1 attack), `type` (renamed `attack_type`) |

Many fields apply only to one protocol, so they are mostly `-`. The SSL, HTTP and weird columns are
99.8–100% `-`, `dns_*` is 83%, and `service` is 63%.

![sparsity](../results/eda/sparsity.png)

## Cleaning (`cyberdefense.data.ton_iot.clean`)
| Step | Why |
|---|---|
| Drop the free-text columns, *then* 20,599 duplicate rows (9.8%) → 190,444 rows | Copies in both train and test inflate scores. Ransomware (26%), xss (24%) and normal (16%) have the most. 30 of the rows (all normal) differ from another row only in a dropped text column, such as the DNS query string. Deduplicating before dropping those columns missed them; validation caught it. |
| Treat `-` as "not applicable": 0 for numeric fields, `"none"` for categorical | Zeek writes `-` when a protocol field doesn't apply. There are no true NaNs. |
| Drop the 5 free-text columns | They act as signatures (for example the literal SQL-injection URI), not behaviour. |
| Treat DNS/HTTP codes and T/F flags as categorical | A DNS qtype of 28 is not "more" than 1. |
| Keep the 4 identifier columns for graph construction only | See [Leakage](#leakage-host-identity-gives-the-label-away). |
| Add `feature_group`, a hash of the 33 feature values | Used by the split to keep repeated feature vectors together. |
| Fail on any unexpected or missing column | A schema change in the source can't pass through silently. |
| Do not scale or encode here | Scalers and encoders are fitted on the training split only (see [Preprocessing](#preprocessing-cyberdefensedatapreprocess)). |

No two exactly identical feature rows had conflicting labels in the raw file. Conflicts appear only once
IPs and ports are ignored; see [Conflicting labels](#conflicting-labels-a-hard-limit-on-accuracy).

## Class balance (after dedup)
| Class | Rows | Share |
|---|---|---|
| normal | 42,010 | 22.1% |
| scanning | 20,000 | 10.5% |
| ddos | 19,993 | 10.5% |
| injection | 19,964 | 10.5% |
| password | 19,861 | 10.4% |
| dos | 18,992 | 10.0% |
| backdoor | 18,711 | 9.8% |
| xss | 15,137 | 7.9% |
| ransomware | 14,735 | 7.7% |
| mitm | 1,041 | 0.5% |

- **Binary:** 148,434 attack vs 42,010 normal (78% attack). Attack is the *majority*, so accuracy is
  misleading. Report F1, precision, recall and **FPR**.
- **Multiclass:** mitm is very rare (0.5%). Use macro-F1, stratified splits and per-class reports.

![class balance](../results/eda/class_balance.png)

## Leakage: host identity gives the label away
The attacking machines (192.168.1.30–39) almost only produce attack traffic.

![src ip purity](../results/eda/src_ip_purity.png)

- 43 of the 51 source IPs are 100% one class (44.5% of flows). Only 3 of the 1,026 host pairs have mixed labels.
- Quick check (random 70/30 stratified split, after dedup):

  | Model | Binary F1 |
  |---|---|
  | Lookup "majority label of this src_ip" | 0.994 |
  | Random Forest, flow features incl. ports | 0.999 |
  | Random Forest, flow features, no ports or IPs | 0.999 |

**Implications**
1. Binary detection on a random split is **saturated**. It can't separate RF, GNN and RL, so it can't answer
   the research question on its own.
2. Host identity must never be a feature. That means no one-hot IPs, no learnable per-node embeddings, and no
   ports as raw features. Otherwise a GNN can learn "this node is an attacker" instead of attack behaviour.

## Repeated feature vectors
Exact-duplicate removal only catches rows that also match on IPs and ports. Once IPs and ports are excluded,
**96,994 rows (50.9%) repeat another row's feature vector exactly**. There are only 93,450 distinct vectors.

| Class | Share of rows that repeat another |
|---|---|
| ransomware | 98.3% |
| backdoor | 89.5% |
| scanning | 77.6% |
| dos | 66.1% |
| normal | 43.7% |
| xss | 42.9% |
| ddos | 40.3% |
| password | 23.2% |
| injection | 0.9% |
| mitm | 0.5% |

These are real repeated traffic (for example the same SMB probe sent to many ports), so they are **not deleted**.
Deleting them would shrink ransomware to about 250 rows and break the graph, because each one is a real edge
between hosts.

The groups are extremely uneven. A single vector, a TCP flow of one SYN packet with no reply (`S0`, 0 bytes),
covers **15,344 rows, 77% of all scanning**. One vector covers 35% of xss, and 12 vectors cover 78% of
ransomware. These are thousands of separate connections that look identical at flow level, and a realistic
test set must contain them.

**Correction.** An earlier version of this doc said a plain random split inflates multiclass macro-F1 by about
14 points (0.953 vs 0.815 for a fully grouped split). That was wrong. A fully grouped split must put each giant
group in one split, which **breaks stratification**: scanning ended up at 3% of train and 38% of test. The
score drop came from that class shift, not from memorisation. Measured with RF on the test split, varying how
large a group may be before it is split row-wise:

| Groups kept intact up to | Worst class-share deviation | Test rows whose vector occurs in train | RF macro-F1 |
|---|---|---|---|
| 1 row (plain stratified) | 0.00 pts | 53.3% | 0.954 |
| 10 rows | 0.00 pts | 44.8% | 0.949 |
| 50 rows | 0.01 pts | 36.9% | 0.953 |
| 200 rows | 0.01 pts | 22.1% | 0.952 |
| no limit (fully grouped) | 27.12 pts | 0.0% | 0.753 |

Memorising repeated vectors does **not** inflate scores here. Grouping is kept as hygiene only, with large
groups allowed to split.

## Conflicting labels: a hard limit on accuracy
169 identical feature vectors carry more than one attack type (16,306 rows: mostly ransomware 8,410,
xss 6,383 and ddos 1,064; common pairs are injection/xss and ddos/injection). Thirteen of them conflict even
on the binary label. No model can separate these from flow features alone. They are kept and treated as
irreducible error. Host context is exactly what a GNN adds, so this is a place where the GNN could win.

## Feature behaviour
Attack types differ clearly in volume features (bytes, packets, duration), which is why tabular models do so well.

![features by class](../results/eda/features_by_class.png)

- **Skew:** the numeric features have skewness 70–420 (bytes range 0 to 3.9 × 10⁹). No negatives or infinities.
- **Near-constant columns:** 15 HTTP, SSL and weird columns are more than 99.7% one value.
- **Zeek quirks:** in 985 rows `src_ip_bytes < src_bytes` (1,669 for the dst side). This is a known Zeek effect
  from sequence-number byte counting.
- **Correlation:** only `dst_pkts` and `dst_ip_bytes` correlate above 0.95 (on log scale).

## Preprocessing (`cyberdefense.data.preprocess`)
Fitted on the **training split only**, then applied to val and test. Every model uses the same fitted transform.
- **Numeric:** `log1p`, then standardise.
- **Categorical:** one-hot encode. Categories seen fewer than 20 times in train, and categories unseen in train,
  map to an "infrequent" column. This gives **103 features** on the current training split.
- **Kept as is:** near-constant columns (their rare values identify web attacks, and the infrequent rule bounds
  their size), both correlated columns, the Zeek quirks, and outliers (attacks are outliers by nature, and the
  log transform tames the scale).

## Train / validation / test split (`cyberdefense.data.splits`)
- **70 / 15 / 15**, via `StratifiedGroupKFold` with 20 folds: 3 folds form test, 3 form validation and 14 form
  train. Seed 42.
- **Stratified on `attack_type`**, which also stratifies the binary label.
- **Grouped on `feature_group`** for groups of up to 200 rows (`max_group_size`). Larger groups are split row-wise.
- The validation split is for model selection and early stopping. Test is touched only for final reporting.

| | train | val | test |
|---|---|---|---|
| rows | 133,310 | 28,567 | 28,567 |
| attack share | 77.94% | 77.94% | 77.94% |
| mitm rows | 728 | 157 | 156 |

Checks, recorded in `metadata.json`:
- Every class has the same share in all three splits (max deviation below 0.01 points).
- No kept-intact group crosses splits, and no full row appears in more than one split.
- 22% of test rows share a feature vector with some training row; these are the large common patterns.

## Imbalance strategy
Use **class weights** in the loss (balanced sample weights for RF and XGBoost), stratified splits and macro
metrics. No SMOTE or other synthetic oversampling: synthetic flows have no real hosts, so they can't be placed in
the graph, and every model must be evaluated on the real distribution.

## Evaluation protocols
| Protocol | What it measures | Status |
|---|---|---|
| `random` | Stratified split as is, binary and multiclass | Implemented (`scripts/train_baselines.py`) |
| `unseen_attack` | For each attack type: remove it from train and val, then measure how much of it is still flagged in test (`holdout_recall`), plus FPR on normal traffic | Implemented, binary task |
| `unseen_host` | For each attacker host with at least 500 attack flows (9 hosts): remove every flow it sends *or* receives from train and val, then measure detection of its test attack flows | Implemented, binary task (docs/03) |

Validation found that **100% of test attack flows come from hosts already seen attacking in training**, so the
random split never tests unfamiliar devices. That is why `unseen_host` exists.

An early one-off RF check (70/30 split, no threshold tuning or class weights) suggested ddos 77%, mitm 88%,
password 89% and injection above 99%. **The formal runs replace it** (docs/02_baselines.md): untuned RF
recall on the held-out type was injection 0.785, password 0.864, mitm 0.885, dos 0.950, ddos 0.980, and
0.986 or higher for the rest. The early check had injection and ddos badly wrong.

## Graph structure
- 776 hosts (51 that send flows, 753 that receive them, 176 IPv6) and only **1,026 unique host pairs** for 190k
  flows. The graph is a few hubs with many repeated flows (multi-edges).
- Degree is extremely skewed: median 3, maximum 60,569 (192.168.1.30).
- **The file has no timestamp column**, so time-windowed or dynamic graphs aren't possible with it. The full
  ToN-IoT network data does include `ts`, if we need time later.

![host degree](../results/eda/host_degree.png)

### Decision: graph design for the GNN stage
**Homogeneous host graph with flows as edges; the task is edge classification** (E-GraphSAGE style).
- Nodes: hosts (IPs). Node features are computed from **training edges only**, such as degree and aggregated
  flow statistics, never the identity.
- Edges: flows, with the 103 preprocessed features as edge attributes. Multi-edges are kept.
- Model: GCN, GraphSAGE or GAT produces host embeddings, then an MLP classifies `[h_src ‖ h_dst ‖ flow_features]`.
  This keeps one comparable model family across all three architectures.
- Message passing during training uses only training edges, so test flows never leak.

Why not the alternatives:
- **Heterogeneous** (host / port / service node types): the only real entity here is the host. Ports carry
  leakage and `service` is 63% "none". Extra node types add complexity without adding information. Keep a
  host–flow bipartite heterogeneous graph as an optional ablation later.
- **Flows as nodes** (flows linked when they share a host): the 60k-degree hub alone would create about
  1.8 × 10⁹ links. Not feasible without heavy sampling.
- **Temporal graphs:** no timestamps in this file.

## Code map
| Module | Role |
|---|---|
| `cyberdefense/data/ton_iot.py` | Schema constants, `clean()`, `load_split()` |
| `cyberdefense/data/splits.py` | Stratified group-aware split and its checks |
| `cyberdefense/data/preprocess.py` | Train-fitted log/scale/one-hot transform |
| `cyberdefense/data/eda.py` | Figures and graph statistics |
| `cyberdefense/data/pipeline.py` | `build_dataset()`: the full chain behind `scripts/prepare_data.py` |
| `tests/test_ton_iot.py`, `test_splits.py`, `test_preprocess.py`, `test_pipeline.py` | Tests on synthetic data |
