"""Train and evaluate the traditional-ML baselines (Random Forest, XGBoost).

Runs every configured model x task x protocol x imbalance strategy x seed on the
prepared splits (run scripts/prepare_data.py first). Usage::

    uv run python scripts/train_baselines.py                     # split set: random
    uv run python scripts/train_baselines.py --dataset.split_set lodo_2018-02-14
    uv run python scripts/train_baselines.py --baseline.protocols '["random"]' \\
        --baseline.imbalance '["none","class_weight","undersample","oversample","smote"]'
    uv run python scripts/train_baselines.py --baseline.seeds '[42,43,44]'
    uv run python scripts/train_baselines.py --baseline.use_tuned   # after tuning
    uv run python scripts/train_baselines.py --baseline.use_tuned --tuning.out_dir tuning

Outputs:
    results/<split_set>/<experiment>/<protocol>/<task>/<run>.json   every metric, confusion
    results/<split_set>/<experiment>/<protocol>/<task>/*__confusion.png
    results/<split_set>/<experiment>/summary.csv                    one row per run
    results/<split_set>/<experiment>/summary_aggregated.csv         mean +- std over seeds
    results/<split_set>/tensorboard/<experiment>/                   curves, confusion, HPARAMS
where <experiment> is baselines, or baselines_tuned[_<suffix>] with --baseline.use_tuned
(suffix from the tuning folder name: tuning_disjoint -> baselines_tuned_disjoint).
"""

import logging

import pandas as pd

from cyberdefense.baselines.experiment import experiment_name, run_all
from cyberdefense.baselines.tuning import apply_tuned
from cyberdefense.config import ExperimentConfig
from cyberdefense.data.schema import SPLITS, load_split
from cyberdefense.reporting import report_tables
from cyberdefense.utils import setup_logging

log = logging.getLogger("train_baselines")


def _report(summary: pd.DataFrame) -> None:
    for title, table in report_tables(summary).items():
        log.info("%s (test):\n%s", title, table.to_string())
    log.info("† = tuning target; its unseen score is optimistic ('tuned for')")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    if cfg.baseline.use_tuned:
        cfg = apply_tuned(cfg)
        log.info("using tuned params from %s", cfg.tuning_dir)
    data = {name: load_split(cfg.dataset, name) for name in SPLITS}
    summary = run_all(cfg, data)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        _report(summary)
    log.info("results -> %s/%s/", cfg.run_dir, experiment_name(cfg))


if __name__ == "__main__":
    main()
