# Adaptive Cyberattack Detection and Automated Defense (GNN + RL)

Research project: does combining Graph Neural Networks and Reinforcement Learning improve network-attack
detection and automated defense compared with traditional ML, standalone GNNs and standalone RL?

## Setup
Requires [uv](https://docs.astral.sh/uv/).
```bash
uv sync
```
Download the ToN-IoT network train/test CSV into `data/raw/ToN-IoT/`:
```bash
mkdir -p data/raw/ToN-IoT
curl -L --fail -o data/raw/ToN-IoT/train_test_network.csv \
  https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/main/train_test_network.csv
```

## Pipeline
The full, ordered protocol is in [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md).
```bash
uv run python scripts/prepare_data.py     # clean → stratified split → parquet → figures
uv run python scripts/validate_data.py    # duplicates, leakage, stratification, provenance
uv run python scripts/tune_baselines.py   # RF random search + XGBoost Optuna TPE (validation only)
uv run python scripts/train_baselines.py --baseline.seeds '[42,43,44,45,46]'
uv run python scripts/train_baselines.py --baseline.use_tuned --baseline.seeds '[42,43,44,45,46]'
uv run python scripts/freeze_baselines.py # snapshot to frozen/ (tracked in git)
uv run python scripts/build_graph.py      # host graph (PyG) + validation
uv run python scripts/train_gnn.py        # MLP, GCN, GraphSAGE, GAT (5 seeds, all protocols)
uv run python scripts/compare_models.py   # GNNs vs frozen baselines -> results/comparison/
uv run python scripts/audit_gnn.py --paths.results_dir results/phase2   # Phase 2 (docs/06)
uv run tensorboard --logdir results/tensorboard
```
Every setting lives in `src/cyberdefense/config.py` and can be overridden on the command line, for example
`--split.seed 7` or `--baseline.models '["xgboost"]'`. Use `--help` to list every option.

## Layout
```
src/cyberdefense/
  config.py          all settings (pydantic)
  data/              ToN-IoT cleaning, splitting, preprocessing, figures, pipeline
  baselines/         Random Forest / XGBoost experiments
  graph/             host-graph construction and validation (PyG)
  gnn/               MLP / GCN / GraphSAGE / GAT edge classifiers
  metrics.py         shared evaluation metrics
  reporting.py       shared result tables (all stages)
  tracking.py        TensorBoard logging
scripts/             entry points
tests/               pytest suite (synthetic data, no download needed)
docs/                research write-ups
frozen/              frozen baseline snapshots (tracked)
data/, results/      generated, gitignored
```

## Development
```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Documentation
- [docs/01_data_exploration.md](docs/01_data_exploration.md): dataset, cleaning, leakage, split, graph design.
- [docs/02_baselines.md](docs/02_baselines.md): baseline method, imbalance handling, metrics, tuning, final 5-seed results and reference scores for the GNNs.
- [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md): final protocol (validation, genuine unseen-attack, unseen-host, 5 seeds, freeze).
- [docs/04_graph_construction.md](docs/04_graph_construction.md): homogeneous host graph (PyG), its validation and what it looks like.
- [docs/05_gnn_experiments.md](docs/05_gnn_experiments.md): Phase 1 GNN benchmark (MLP, GCN, GraphSAGE, GAT): design, training, leakage guards.
- [docs/06_gnn_validation.md](docs/06_gnn_validation.md): Phase 2 leakage audit, host-reputation shortcut, ablations, unseen host-pair protocol.
- [docs/07_unseen_attack_host.md](docs/07_unseen_attack_host.md): Phase 3 unseen attack from unseen hosts: feasibility, shortcut exposure, matched-FPR evaluation.
- [docs/08_dataset_suitability.md](docs/08_dataset_suitability.md): Phase 4 dataset survey (UNSW-NB15, CIC-IDS2017/2018, NetFlow v3, LANL), decision rules, graph strategy.

## Known issue (macOS)
XGBoost and torch each bring their own OpenMP runtime, and they can't both run parallel code in one process:
- if torch loads first, XGBoost training segfaults;
- if XGBoost loads first, multithreaded torch operations segfault.

How the project handles it:
- `cyberdefense.baselines` loads XGBoost first and uses torch only for TensorBoard.
- The graph and GNN code (`cyberdefense.graph`, `cyberdefense.gnn`) never imports XGBoost. Run baselines and GNNs in separate
  processes, which every script already does.
- In a notebook, don't mix the two. If you must, import `cyberdefense.baselines` first and call
  `torch.set_num_threads(1)`.
