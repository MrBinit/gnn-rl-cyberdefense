"""Build the NetFlow v3 dataset (default NF-CICIDS2018-v3): clean -> sample -> split sets.

Reads data/raw/nf/NF-CICIDS2018-v3.csv (20M flows, 3.9 GB) in chunks, drops 27 Feb and
duplicates, computes window features on the complete traffic, keeps whole 5-min windows
(every window with a clean attack + 0.5x attack-free windows, attacks capped per window and
type), then builds every split set (random, lodo_<day>, bot, bot_unseen_hosts). Usage::

    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --no-dataset.make_figures

Outputs:
    data/processed/nf_cicids2018/clean.parquet, splits.parquet, metadata.json
    results/eda/*.png, summary.json
Next: scripts/validate_data.py, then scripts/freeze_splits.py.
"""

import logging

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.utils import setup_logging

log = logging.getLogger("prepare_data")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    meta = build_dataset(cfg)

    checks = meta["random_checks"]
    log.info("days: %s; noisy-label rows: %s", len(meta["days"]), meta["noisy_label_rows"])
    log.info("bot hosts: %s", meta["bot_hosts"])
    for name, s in meta["split_sets"].items():
        log.info("%-24s rows=%s excluded=%s", name, s["rows"], s["excluded_rows"])
    log.info("random: max class-share deviation %.5f", checks["max_class_share_deviation"])
    log.info("random: feature-group overlap %s", checks["feature_group_overlap"])


if __name__ == "__main__":
    main()
