"""Compare the GNN results with the frozen baselines (no training, reads summary.csv files).

Usage::

    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --dataset.split_set bot_unseen_hosts
    uv run python scripts/compare_models.py --gnn.compare_with \\
        '["frozen/baselines_nf_cicids2018_v1/random/results/baselines/summary.csv"]'
    uv run python scripts/compare_models.py --graph.message_passing transductive

Reads ``--gnn.compare_with`` (default: every frozen baseline summary of the split set,
frozen/baselines_<tag>/<split_set>/results/*/summary.csv) plus
results/<split_set>/<gnn experiment>/summary.csv (if it exists) and writes
results/<split_set>/comparison/<name>.md (headline + per-holdout tables)
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
    own = cfg.run_dir / experiment_name(cfg) / "summary.csv"
    paths = list(dict.fromkeys([*cfg.baseline_summaries, *([own] if own.exists() else [])]))
    name = cfg.gnn.report_name or experiment_name(cfg)
    for p in paths:
        log.info("reading %s", p)
    summary = load_summaries(paths)
    out = cfg.run_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    headline = headline_table(summary, cfg.tuning.holdouts)
    headline.to_csv(out / f"{name}__headline.csv")
    md = comparison_markdown(summary, cfg.tuning.holdouts, f"Baselines vs {name}")
    (out / f"{name}.md").write_text(md)
    print(md)
    log.info("comparison -> %s/%s.md", out, name)


if __name__ == "__main__":
    main()
