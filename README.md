# Adaptive Cyberattack Detection and Automated Defense (GNN + RL)

Research project: does combining Graph Neural Networks and Reinforcement Learning improve network-attack
detection and automated defense compared with traditional ML, standalone GNNs and standalone RL?

## Setup
Requires [uv](https://docs.astral.sh/uv/).
```bash
uv sync
```
Download **NF-CICIDS2018-v3** (UQ NetFlow v3 datasets, https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
and put the CSV at `data/raw/nf/NF-CICIDS2018-v3.csv` (20M flows, 3.9 GB). See docs/09.

## Pipeline
The full, ordered protocol is in [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md).
```bash
uv run python scripts/prepare_data.py     # chunked read → dedup → window sample → split sets → parquet
uv run python scripts/validate_data.py    # provenance, preparation (windows), every split set
uv run python scripts/freeze_splits.py    # frozen/data_<tag>/: split-set fingerprints (verify_splits)
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
  data/              NetFlow v3 loading/cleaning (netflow.py), schema, splits, preprocessing,
                     holdout protocols, validation, figures, dataset survey
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
Doc 01 explores the current dataset (NF-CICIDS2018-v3); 02–07 record Phases 1–3 on the first dataset (ToN-IoT,
since removed); 08 chooses the new dataset; 09 is the current pipeline.

- [docs/01_data_exploration.md](docs/01_data_exploration.md): full EDA of NF-CICIDS2018-v3 (time, hosts, label quality, features, window graphs) and the resulting data/graph/model decisions.
- [docs/02_baselines.md](docs/02_baselines.md): baseline method, imbalance handling, metrics, tuning, final 5-seed results and reference scores for the GNNs.
- [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md): final protocol (validation, genuine unseen-attack, unseen-host, 5 seeds, freeze).
- [docs/04_graph_construction.md](docs/04_graph_construction.md): homogeneous host graph (PyG), its validation and what it looks like.
- [docs/05_gnn_experiments.md](docs/05_gnn_experiments.md): Phase 1 GNN benchmark (MLP, GCN, GraphSAGE, GAT): design, training, leakage guards.
- [docs/06_gnn_validation.md](docs/06_gnn_validation.md): Phase 2 leakage audit, host-reputation shortcut, ablations, unseen host-pair protocol.
- [docs/07_unseen_attack_host.md](docs/07_unseen_attack_host.md): Phase 3 unseen attack from unseen hosts: feasibility, shortcut exposure, matched-FPR evaluation.
- [docs/08_dataset_suitability.md](docs/08_dataset_suitability.md): Phase 4 dataset survey (UNSW-NB15, CIC-IDS2017/2018, NetFlow v3, LANL), decision rules, graph strategy.
- [docs/09_nf_cicids2018_pipeline.md](docs/09_nf_cicids2018_pipeline.md): NF-CICIDS2018-v3 pipeline: feature policy, sampling, run order.

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
