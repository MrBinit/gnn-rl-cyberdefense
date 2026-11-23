# Adaptive Cyberattack Detection & Automated Defense (GNN + RL)

Research project: does GNN + RL beat traditional ML, standalone GNN, and standalone RL
for network attack detection and automated defense?

## Plan
1. Data: ToN-IoT (detection), Cyberwheel (simulated defense environment) — ToN-IoT done
2. Baselines: Random Forest, XGBoost — final, frozen in `frozen/baselines_v2` (results docs/02,
   protocol docs/03). Headline = recall at 0.25%/1% FPR, thresholds from val (`metrics.fixed_fpr_metrics`).
3. GNNs: GCN, GraphSAGE, GAT (host graph, flows as edges, edge classification) — graph built
   and validated (docs/04); Phase 1 (MLP control + GCN/SAGE/GAT, shared edge head) implemented,
   run (docs/05). Phase 2 (docs/06): no leakage, but gains = host-profile shortcut (shuffled
   structure ≈ no message passing ≈ GNN); RF best on unseen_host. Next: host-disjoint protocols,
   shortcut-free node inputs / flow graph, or Cyberwheel. Phase 3 (docs/07): protocol
   unseen_attack_host (type + all its senders held out), matched-actual-FPR recall + Wilson CI.
   Result: GNNs tie RF on clean types (scanning easy, mitm fails all), win only on
   dst-reputation types (= no_propagation), worse on novel-endpoint; 8-10x RF's FPR at val
   thresholds. No evidence of GNN generalisation on ToN-IoT. Phase 4 (docs/08): dataset survey
   tool (`scripts/survey_dataset.py`). NF-UNSW-NB15-v3 rejected (44 hosts, 4 attackers, src F1 1.0).
   NF-CSE-CIC-IDS2018-v3 chosen, conditionally (205k hosts, src F1 0.92, timestamps): mixed-role
   attackers only in Bot/Infiltration (relational headline); external DoS/DDoS/brute force are
   dedicated single hosts hitting known victims; one attack family per day; noisy infiltration/web labels
4. RL: PPO, DQN
5. GNN + RL: GNN graph embeddings as the RL policy's state
6. Evaluation: F1, precision, recall, FPR, attack mitigation success, defense cost,
   generalization to unseen attacks
7. Interactive dashboard for threats and responses

## Commands (order: docs/03_evaluation_protocol.md)
- `scripts/prepare_data.py` → `data/processed/ton_iot/{clean,train,val,test}.parquet`
- `scripts/validate_data.py` → `results/validation/report.json` (must pass before freezing)
- `scripts/tune_baselines.py` → `results/tuning_disjoint/<model>/` (holdouts xss/dos/ddos)
- `scripts/train_baselines.py [--baseline.use_tuned] --baseline.seeds '[42,...,46]'`
  → `results/baselines/` or `results/baselines_tuned_disjoint/`
- `scripts/freeze_baselines.py` → `frozen/baselines_<tag>/` (tracked in git)
- `scripts/build_graph.py` → `data/processed/graph/<protocol>.pt` + `results/graph/` (validated; docs/04)
- `scripts/train_gnn.py --gnn.device mps [--gnn.resume]` → `results/gnn/` (5 seeds, all protocols; MPS = Apple GPU, 2–4x faster)
- `scripts/compare_models.py` → `results/comparison/gnn.md` (vs frozen baselines)
- Phase 2: `scripts/audit_gnn.py`, `train_gnn.py --gnn.ablation <a>`, protocol `unseen_pair`,
  model `host_prior`; all outputs under `--paths.results_dir results/phase2` (run order docs/06 §6)
- Phase 4: `scripts/survey_dataset.py --survey.dataset <netflow_v3|cic_ids2017|unsw_nb15|lanl>`
  → `results/datasets/comparison.csv`; decision rules in docs/08 §4
- Phase 3: `--baseline/gnn.protocols '["unseen_attack_host"]'` under `results/phase3` (docs/07 §5).
  Holdout names carry the host set (`type@ip,ip`); evaluate with `target_mask`, remove with `holdout_mask`
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format .` — keep both clean.
- Bool CLI options are flags: `--baseline.use_tuned`, `--no-ton_iot.make_figures` (no `true`).

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

## Data facts that constrain every model (details: docs/01_data_exploration.md)
- src_ip alone predicts the label (F1 0.994): never use IPs/ports as features or
  per-node learnable embeddings.
- Host reputation: 95–100% of every held-out attack type's test flows come from hosts that
  attack in training, and 99.8% of test flows use a host pair seen in training. Label-only
  `host_prior` beats RF on mitm. Graph models must be judged on unseen_host / unseen_pair (docs/06).
- Only 19 attacker hosts; .30/.31 send 6 of 9 types. With attackers held out, shortcuts remain:
  destination reputation (ddos/injection/password) and novel endpoint (backdoor/ransomware:
  "both ends unseen -> attack"). scanning is the only clean unseen-attack+host test (docs/07).
- Binary detection on a random split is saturated (F1 ≈ 0.997); the comparison rests on
  genuine unseen-attack (types not used in tuning), unseen-host and multiclass macro-F1.
- GNN stage must call `cyberdefense.freeze.verify_splits` against the frozen manifest and
  keep held-out hosts out of the training graph.
- 51% of flows repeat another flow's feature vector; the split keeps groups ≤200 rows together.
- No timestamps in this file, so no temporal graphs.
- Imbalance: 78% attack, mitm 0.5%. Default handling = class-weighted loss + (binary)
  threshold tuned on val; never compare on accuracy (see docs/02_baselines.md).

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
