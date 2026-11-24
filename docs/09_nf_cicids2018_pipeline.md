# 09 · NF-CICIDS2018-v3 pipeline (Phase 5 setup)

The project now runs on **NF-CICIDS2018-v3**, chosen in docs/08 §5b. ToN-IoT data, results and frozen baselines
were removed. Docs 01–07 remain as the written record of Phases 1–3 on ToN-IoT, and the frozen ToN-IoT baselines
are in git history (first commit).

## Data
- **File:** `data/raw/nf/NF-CICIDS2018-v3.csv`: 20,115,529 flows, 55 columns, 16 days (14 Feb – 2 Mar 2018), 14
  attack types. Source: UQ NetFlow v3 (docs/08 §6).
- **Loader:** `cyberdefense.data.netflow`. It reads in chunks with compact dtypes (about 5 GB for the full file).
  Any NetFlow v3 file (NF-UNSW-NB15-v3, NF-ToN-IoT-v3…) loads the same way, which enables the cross-dataset test.

### Feature policy (from the exploration, docs/01 §7)
| Columns | Use | Why |
|---|---|---|
| IPs, ports | graph ids only, **never features** | host identity is a label shortcut |
| `FLOW_START/END_MILLISECONDS` | `timestamp`, `day`, `window` (splits and graphs) only | one attack family per day: time is a shortcut |
| `DNS_QUERY_ID` | dropped | random transaction id |
| `MIN_TTL`, `MAX_TTL` | **off by default** (`--dataset.use_ttl_features`) | each attack tool has one fixed TTL: a machine fingerprint |
| `FTP_COMMAND_RET_CODE` | **removed** | constant |
| `LONGEST_FLOW_PKT` | **removed** | identical to `MAX_IP_PKT_LEN` (r = 1.0) |
| `RETRANSMITTED_IN/OUT_PKTS` | **removed** | duplicates of the byte counts (r ≥ 0.998) |
| `ICMP_TYPE` | **removed** | type×256+code; `ICMP_IPV4_TYPE` is kept |
| 32 counts, durations, throughputs, IATs | numeric: log1p (clipped at 0) and standardised, fitted on train | |
| 7 protocol / flag / type codes | categorical: one-hot, rare categories collapsed | codes, not quantities |
| **8 window features** (`WIN_*` counts, `SRC_INTERNAL`, `DST_INTERNAL`) | always stored; model inputs only with `--dataset.use_window_features` | per 5-min window structure, computed on the **complete** traffic before sampling (docs/01 §8). The fair tabular baseline for GNNs uses them |

That gives 39 flow features, or 47 with the window features.

### Cleaning, windows and sampling (`cyberdefense.data.netflow.clean`)
1. **Drop excluded days:** `exclude_days` = 27 Feb (828 flows).
2. **Remove exact duplicates:** compared on every column the prepared table keeps (ids, start time, features,
   labels). Rows that differ only in dropped columns would otherwise become copies, some of them across splits.
3. **Window features on the complete deduplicated traffic.** Every row also gets `day` (YYYY-MM-DD) and
   `window` (flow start // `window_seconds`, default 300 s).
4. **Window sampling (`sampling=windows`, default).** It keeps **whole windows**, so graph neighbourhoods survive:
   - every window with a reliably labelled (non-noisy) attack;
   - `normal_window_ratio` (0.5) × as many attack-free windows, chosen at random (seeded);
   - within the kept windows, at most `max_attack_flows_per_window` (5,000) flows per attack type, which thins
     the DDoS bursts; normal traffic stays whole.

   `sampling=per_class` (random caps of 500k normal / 50k per type) remains for tabular-only experiments. It
   breaks graphs (docs/01 §9.1).
5. **`noisy_label` = 1** for `infilteration`, `brute_force_-web`, `brute_force_-xss` and `sql_injection`. Their
   labels are mostly normal traffic marked by time window (docs/01 §6). They stay in the data but are out of
   headline results.
6. **`feature_group`:** a hash of the feature vector, so identical flows stay in one split.

### Split sets (`cyberdefense.data.splits.make_split_sets`)
One prepared table (`clean.parquet`) plus `splits.parquet`, which is row-aligned with it and has one column per
split set holding `train` / `val` / `test` / `excluded`. `load_split(cfg.dataset, "train")` returns the split of
the configured set (`--dataset.split_set`, default `random`) and drops `excluded` rows. Callers do not change.

| Split set | Train / val / test | Excluded | What it tests |
|---|---|---|---|
| `random` | 70/15/15, stratified by attack type, grouped by feature vector | – | **sanity check only:** 100% of test attack flows come from hosts seen attacking in train |
| `lodo_<day>` (6 sets: 14, 15, 16, 20, 21 Feb, 2 Mar) | test = that day, val = the next clean-attack day (cyclic), train = every other day, including the normal-only and noisy-only days | – | an **unseen attack family** in an unseen time period |
| `bot` | test = 2 Mar, val = 21 Feb, train = the rest | 26,043 rows touching the C&C `18.219.211.138` in train/val (it ran GoldenEye on 15 Feb) | bot detection without the C&C's history |
| `bot_unseen_hosts` | as `bot` | + 80,443 rows touching the 8 bot hosts in train/val (106,486 in total) | **headline relational test:** infected hosts never seen in training |

Rows per split on the real data:

| Split set | train | val | test | test attacks |
|---|---:|---:|---:|---|
| `random` | 3,129,367 | 670,642 | 670,568 | all 10 clean types |
| `lodo_2018-02-14` | 3,251,217 | 414,774 | 804,586 | ftp-, ssh-bruteforce |
| `lodo_2018-02-15` | 3,686,586 | 369,217 | 414,774 | dos goldeneye, slowloris |
| `lodo_2018-02-16` | 3,696,733 | 404,627 | 369,217 | dos hulk, slowhttptest |
| `lodo_2018-02-20` | 3,569,440 | 496,510 | 404,627 | ddos loic-http |
| `lodo_2018-02-21` | 2,636,867 | 1,337,200 | 496,510 | ddos hoic, loic-udp |
| `lodo_2018-03-02` | 2,328,791 | 804,586 | 1,337,200 | bot |
| `bot` | 2,610,824 | 496,510 | 1,337,200 | bot |
| `bot_unseen_hosts` | 2,564,102 | 462,789 | 1,337,200 | bot |

Settings: `--split.lodo_days` (empty = every clean-attack day), `--split.bot_attack_type`, `--split.bot_val_day`
(None = the last clean-attack day before the bot day), `--split.bot_cnc_hosts`. The random set uses
`n_folds`/`val_folds`/`test_folds`/`max_group_size`/`seed` as before. The split is vectorised; scikit-learn's
`StratifiedGroupKFold` was too slow on millions of groups.

**Caveats found while validating the split sets** (share of test attack flows whose source attacked in train):
- **20 and 21 Feb use the same 10 attacker hosts** (18.x AWS: LOIC-HTTP on the 20th, HOIC/LOIC-UDP on the 21st).
  `lodo_2018-02-21` is 100% "seen attackers": it tests a new attack tool from known hosts. `lodo_2018-02-20` is
  10% in train, and its val day (21 Feb) has the same attackers. Report these two days separately from the other
  four LODO days (0% seen attackers), and never claim host generalisation from them.
- **5 of the 8 bot hosts carry noisy `infilteration` labels in train** (time-window labelling on 28 Feb–1 Mar),
  so `bot` and `lodo_2018-03-02` are 60% "seen attackers". Host reputation leaks through the noisy labels;
  `bot_unseen_hosts` (0%) removes it and is the headline.
- Noisy-label rows (9,641) are in train for every set. Training on them is a choice per model (dropping them from
  train is a reasonable default); they are never in headline test metrics.

### Validation (`scripts/validate_data.py`, all hard checks)
- **Provenance:** raw class counts equal the published NF-CICIDS2018-v3 counts (20,115,529 rows).
- **Cleaning and feature selection:** `clean.parquet` is reproducible from the raw file (row hashes identical, in
  order). No identifier, time, TTL or removed column is among the 39 features. No non-finite values.
  `noisy_label` matches the noisy types. No exact duplicates.
- **Window sampling:** every row's `window` matches its timestamp. The raw rows of the 270 kept windows (180 with
  clean attacks) are re-cleaned without sampling: every kept window holds its **complete** deduplicated normal
  traffic; every (window, attack type) holds min(complete, 5,000) attack flows; every prepared row, including its
  window features, matches that complete-traffic table exactly.
- **Every split set:** train/val/test non-empty, clean attacks in test, no row in two splits; in the bot sets no
  train/val row touches a masked host. Random set: no feature group crosses splits, and the class-share deviation
  is 0.00007 (limit 0.01).

### Freeze (`scripts/freeze_splits.py` → `frozen/data_nf_cicids2018_v1/`, tracked in git)
`manifest.json` holds the raw-file sha256, sha256 of `clean.parquet` and `splits.parquet`, a content fingerprint
(rows + sha256 of the row hashes) of every split of every split set, the dataset and split config, the environment
and the git commit. `data_metadata.json` and `validation_report.json` are copied next to it.
`verify_splits(cfg, manifest)` checks the configured split set against it. `build_graph`, `train_gnn` and
`audit_gnn` call it by default (`GraphConfig.manifest`). The baseline freeze also records its `split_set`.

## Run order
```bash
uv run python scripts/prepare_data.py       # -> data/processed/nf_cicids2018/{clean,splits}.parquet + metadata
uv run python scripts/validate_data.py      # provenance, preparation, every split set (must PASS)
uv run python scripts/freeze_splits.py      # -> frozen/data_nf_cicids2018_v1/manifest.json
uv run python scripts/survey_dataset.py     # optional: shortcut survey of the prepared splits (docs/08)
# then per split set (--dataset.split_set random | lodo_<day> | bot | bot_unseen_hosts),
# each writing to results/<split_set>/ and frozen/baselines_<tag>/<split_set>/:
uv run python scripts/tune_baselines.py
uv run python scripts/train_baselines.py --baseline.seeds '[42,43,44,45,46]'
uv run python scripts/freeze_baselines.py   # -> frozen/baselines_nf_cicids2018_v1/<split_set>/
uv run python scripts/build_graph.py        # verify_splits against the frozen data manifest
```

## What changed in the code
- `data/schema.py`: shared column names and split loading (was part of `ton_iot.py`).
- `data/netflow.py`: NetFlow v3 schema, chunked loading, cleaning, sampling and the feature policy (replaces
  `ton_iot.py`).
- `DatasetConfig` (`--dataset.*`) replaces `TonIoTConfig` (`--ton_iot.*`).
- Validation compares raw class counts with the published file. The EDA adds `classes_per_day.png`.
- Tuning holdouts are now `dos_attacks-goldeneye`, `ddos_attack-loic-udp` and `ftp-bruteforce`, which keeps
  bot/infiltration (the relational test cases) genuinely unseen.
- Default frozen tag `nf_cicids2018_v1`: data in `frozen/data_nf_cicids2018_v1/` (the manifest the GNN stage
  verifies against), baselines in `frozen/baselines_nf_cicids2018_v1/`.
- `--split.strategy` / `by_day` were replaced by the split sets; `train/val/test.parquet` by `splits.parquet`.
- Graph validation counts numeric features from the feature names and ignores columns that are constant in train.
- `.gitignore`: `/data/` and `/results/` are now anchored to the repo root. The old unanchored `data/` rule also
  excluded the code package `src/cyberdefense/data/` from git.

### Per-split-set outputs
Every model stage reads and writes inside the folder of its split set (`--dataset.split_set`, default `random`),
so runs on different sets never overwrite each other. The paths are decided in one place, the `ExperimentConfig`
properties (`run_dir`, `run_tb_dir`, `tuning_dir`, `graph_dir`, `graph_report_dir`, `frozen_baselines_dir`):

```
results/<split_set>/baselines/, baselines_tuned_disjoint/   summaries + per-run JSON
results/<split_set>/tuning_disjoint/<model>/                tuning trials, best_params.json
results/<split_set>/tensorboard/                            all curves of this split set
results/<split_set>/gnn*/, graph/, comparison/, audit/
data/processed/graph/<split_set>/                           saved graphs
frozen/baselines_<tag>/<split_set>/                         frozen baselines (compare_models default)
frozen/data_<tag>/                                          data + every split set (shared)
results/validation/, results/eda/, results/datasets/        data-level, shared
```
Relative `--paths.tb_dir`, `--tuning.out_dir` and `--graph.report_dir` are resolved inside
`results/<split_set>/`; absolute paths are used as given. `--gnn.compare_with` defaults to every frozen
baseline summary of the split set.

## Not done yet (next Phase 5 steps)
- **Baselines on the split sets,** with and without `--dataset.use_window_features` (the fair GNN baseline).
- **Time-windowed graph builder** using `window` and the `WIN_*` features (docs/01 §9.3).
- **Cross-dataset evaluation** on NF-UNSW-NB15-v3.
