# Adaptive Cyberattack Detection & Automated Defense (GNN + RL)

Research project: does GNN + RL beat traditional ML, standalone GNN, and standalone RL
for network attack detection and automated defense?

## Plan
1. Data: **NF-CICIDS2018-v3** (UQ NetFlow v3; chosen in docs/08 §5b, pipeline docs/09);
   Cyberwheel later for the defense stage. ToN-IoT (phases 1-4, docs/01-08) was dropped: its
   data/results are deleted, frozen ToN-IoT baselines live in git history only.
2. Baselines: Random Forest, XGBoost. Headline = recall at 0.25%/1% FPR, thresholds from val
   (`metrics.fixed_fpr_metrics`), plus matched actual-FPR recall + Wilson CI (docs/07).
3. GNNs: MLP control + GCN/SAGE/GAT with shared edge head, `host_prior` and no_propagation
   references, leakage audit (docs/05-07). Lesson from ToN-IoT: gains came from host
   reputation, not structure; judge GNNs only on host-disjoint tests.
4. RL: PPO, DQN
5. GNN + RL: GNN graph embeddings as the RL policy's state
6. Evaluation: F1, precision, recall, FPR, attack mitigation success, defense cost,
   generalization to unseen attacks
7. Interactive dashboard for threats and responses
Phase 5: data prepared, validated and frozen with split sets random / lodo_<day> / bot /
bot_unseen_hosts (docs/09); every model output lives in results/<split_set>/ (config properties
`run_dir`, `tuning_dir`, ...; never build result paths from `paths.results_dir`). Next: baselines
on the split sets with/without window features, time-windowed graph
with window-local node features, baselines with the same per-window host aggregates,
cross-dataset test on NF-UNSW-NB15-v3 (docs/08 §5, §7; docs/09 "Not done yet").

## Commands (order: docs/09, then docs/03)
- `scripts/prepare_data.py` → `data/processed/nf_cicids2018/{clean,train,val,test}.parquet`
  (chunked read of `data/raw/nf/NF-CICIDS2018-v3.csv`, drop 27 Feb, dedup, window features on
  complete traffic, window sampling: whole 5-min windows with clean attacks + 0.5x attack-free
  windows, attack flows capped 5k per (window, type)) + `splits.parquet` (one column per split
  set; `load_split` follows `--dataset.split_set`, default `random`, and drops `excluded` rows)
- `scripts/validate_data.py` → `results/validation/report.json` (must pass before freezing;
  re-checks cleaning, feature policy, window completeness/cap and every split set)
- `scripts/freeze_splits.py` → `frozen/data_nf_cicids2018_v1/manifest.json` (verify_splits target)
- `scripts/tune_baselines.py` → `results/<split_set>/tuning_disjoint/<model>/`
  (holdouts dos_attacks-goldeneye / ddos_attack-loic-udp / ftp-bruteforce)
- `scripts/train_baselines.py [--baseline.use_tuned] --baseline.seeds '[42,...,46]'`
- `scripts/freeze_baselines.py` → `frozen/baselines_nf_cicids2018_v1/<split_set>/` (tracked in git)
- `scripts/build_graph.py`, `scripts/train_gnn.py --gnn.device mps`, `scripts/audit_gnn.py`,
  `scripts/compare_models.py` (protocols: random, unseen_attack, unseen_host, unseen_pair,
  unseen_attack_host; ablations via `--gnn.ablation`)
- `scripts/survey_dataset.py [--survey.dataset prepared|netflow_v3|...]` → shortcut survey
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format .` — keep both clean.
- Bool CLI options are flags: `--baseline.use_tuned`, `--no-dataset.make_figures` (no `true`).

## Conventions
- `uv` for everything. Code in `src/cyberdefense/` (import as `cyberdefense.*`); scripts in
  `scripts/` are thin entry points that call library functions.
- All settings are pydantic models in `src/cyberdefense/config.py`. Scripts build
  `ExperimentConfig(_cli_parse_args=True)`; library code and tests never parse argv.
- New library code gets pytest tests in `tests/`, using the synthetic data in `tests/conftest.py`.
- Fair comparisons: same seed, same splits (`load_split`), same features
  (`cyberdefense.data.preprocess`, fitted on train only) and same metrics
  (`cyberdefense.metrics`, 1 = attack) for every model.
- Model selection and tuning on val; test is only for final reporting. Tuning for an
  attack type makes its unseen-attack score 'tuned for' (optimistic); GNNs must be tuned
  with the same protocol.
- Training logs to TensorBoard via `cyberdefense.tracking` (`<metric>/<split>` tags).
- Research findings go in `docs/NN_*.md`.

## Data facts that constrain every model (details: docs/01 EDA, docs/08 §5b, docs/09)
- Never use IPs, ports, flow timestamps, DNS_QUERY_ID or (by default) TTL as features, and no
  per-node learnable embeddings: host identity and time are label shortcuts. Ids, `timestamp`,
  `day`, `window` stay in the prepared tables for graphs, windows and splits only. Removed as
  useless: FTP_COMMAND_RET_CODE, LONGEST_FLOW_PKT, RETRANSMITTED_*_PKTS, ICMP_TYPE.
- One attack family per day (14 Feb FTP/SSH brute force ... 2 Mar bot); only 5% of normal
  flows fall in an attack's time window → a day split = an unseen-attack split; time-of-day
  features would leak.
- External DoS/DDoS/brute-force attackers (18.x/13.x AWS hosts) send 100% attack traffic and
  hit known victim servers: host/victim-confounded, no relational claim rests on them.
  Mixed-role (attackers that also act normally) only in bot (8 internal hosts) and
  infiltration → the headline relational test.
- Source-host majority F1 0.92, pair F1 0.97, 97% of test pairs seen in train (random split).
- Unreliable labels (time-window labelling): xss 91%, sql 78%, infilteration 76%, web brute
  force 68% of their flows share feature vectors with normal traffic → out of the headline.
- 464 internal hosts (172.31.x, stable across days) + 205k external IPs (77% seen one day);
  hubs: DNS 172.31.0.2 (7.3M flows), metadata 169.254.169.254, 0.0.0.0 → inductive graphs,
  neighbour caps. Bot = 8 internal hosts, C&C 18.219.211.138 (= GoldenEye attacker 15 Feb).
- 5-min window graph: median 15k flows / 1.5k hosts; window structure (pair flow count) is
  strong signal (bot AUC 0.96) → baselines must get the same window aggregates.
- Random normal-flow sampling destroys graph neighbourhoods: graph experiments need complete
  traffic of selected windows/days (docs/01 §9.1).
- Split-set caveats: 20/21 Feb DDoS share the same 10 attackers (lodo_2018-02-21 = known hosts);
  5 of 8 bot hosts carry noisy infilteration labels in train → headline is bot_unseen_hosts.
- GNN stage must call `cyberdefense.freeze.verify_splits` against the frozen manifest and
  keep held-out hosts out of the training graph.
- Imbalance: default handling = class-weighted loss + (binary) threshold tuned on val; never
  compare on accuracy.

## Gotcha
- macOS OpenMP clash: xgboost and torch can't both run parallel code in one process
  (torch first → XGBoost segfaults; xgboost first → multithreaded torch segfaults).
  `cyberdefense.baselines` imports xgboost first; `cyberdefense.graph`/GNN code must never import
  xgboost or `cyberdefense.baselines` (shared holdouts live in `cyberdefense.data.protocols`).
  Tests mix both: `tests/conftest.py` loads xgboost first and sets torch to 1 thread.
- GNN layers take `message_passing_graph(data)` (mp_mask edges), never the full edge_index.
- Result tables live in `cyberdefense.reporting` (no xgboost/torch) and metrics in `cyberdefense.metrics`
  (`f1_threshold` = the baselines' threshold tuner, `holdout_metrics`); every stage's per-run JSON
  uses the same keys so one table compares all models.
