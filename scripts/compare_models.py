"""Compare the GNN results with the frozen baselines (no training, reads summary.csv files).

Usage::

    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --gnn.compare_with \\
        '["frozen/baselines_v2/results/baselines/summary.csv",
          "frozen/baselines_v2/results/baselines_tuned_disjoint/summary.csv"]'
    uv run python scripts/compare_models.py --graph.message_passing transductive

Reads every file in ``--gnn.compare_with`` plus <results_dir>/<gnn experiment>/summary.csv
(if it exists) and writes <results_dir>/comparison/<name>.md (headline + per-holdout tables)
and <name>__headline.csv, where <name> is --gnn.report_name or the GNN experiment name.
Model labels get the folder as a suffix, e.g. "sage [shuffled_structure]"
(see reporting.label_runs), so ablations and baselines share one table.
"""

import logging

from cyberdefense.config import ExperimentConfig
from cyberdefense.gnn.experiment import experiment_name
from cyberdefense.reporting import comparison_markdown, headline_table, load_summaries
from cyberdefense.utils import setup_logging

log = logging.getLogger("compare_models")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    own = cfg.paths.results_dir / experiment_name(cfg) / "summary.csv"
    paths = list(dict.fromkeys([*cfg.gnn.compare_with, *([own] if own.exists() else [])]))
    name = cfg.gnn.report_name or experiment_name(cfg)
    for p in paths:
        log.info("reading %s", p)
    summary = load_summaries(paths)
    out = cfg.paths.results_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    headline = headline_table(summary, cfg.tuning.holdouts)
    headline.to_csv(out / f"{name}__headline.csv")
    md = comparison_markdown(summary, cfg.tuning.holdouts, f"Baselines vs {name}")
    (out / f"{name}.md").write_text(md)
    print(md)
    log.info("comparison -> %s/%s.md", out, name)


if __name__ == "__main__":
    main()
