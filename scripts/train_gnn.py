"""Train and evaluate the edge-classification GNNs (MLP control, GCN, GraphSAGE, GAT).

Runs every configured model x task x protocol x holdout x seed on the host graph, with
the frozen splits (checked against the manifest first) and the baselines' evaluation.
Run scripts/build_graph.py once first to confirm the graph validates. Usage::

    uv run python scripts/train_gnn.py                                  # everything, 5 seeds
    uv run python scripts/train_gnn.py --gnn.protocols '["random"]' --gnn.seeds '[42]'
    uv run python scripts/train_gnn.py --gnn.models '["mlp","sage"]'
    uv run python scripts/train_gnn.py --gnn.resume                     # after an interruption
    uv run python scripts/train_gnn.py --graph.message_passing transductive   # -> gnn_transductive

Outputs:
    results/<experiment>/<protocol>/<task>/<run>.json   metrics, confusion, epoch history
    results/<experiment>/summary.csv, summary_aggregated.csv
    results/tensorboard/<experiment>/                   loss/train, loss/val, final metrics
where <experiment> is gnn, or gnn_<message_passing>_<node_features> for non-defaults.
Then compare with the frozen baselines: scripts/compare_models.py.
"""

import logging

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.ton_iot import SPLITS, load_split
from cyberdefense.freeze import verify_splits
from cyberdefense.gnn.experiment import experiment_name, run_all
from cyberdefense.reporting import headline_table
from cyberdefense.utils import setup_logging

log = logging.getLogger("train_gnn")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    verify_splits(cfg, cfg.graph.manifest)
    log.info("splits match the frozen manifest %s", cfg.graph.manifest)
    splits = {name: load_split(cfg.ton_iot, name) for name in SPLITS}
    summary = run_all(cfg, splits)
    if not summary.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            log.info("headline (test):\n%s", headline_table(summary, cfg.tuning.holdouts).T)
    log.info("results -> %s/%s/", cfg.paths.results_dir, experiment_name(cfg))


if __name__ == "__main__":
    main()
