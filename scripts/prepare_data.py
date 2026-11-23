"""Build the ToN-IoT dataset: clean -> stratified split -> parquet -> exploration figures.

Usage::

    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --no-ton_iot.make_figures

Outputs:
    data/processed/ton_iot/{clean,train,val,test}.parquet, metadata.json
    results/eda/*.png, summary.json
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

    checks = meta["checks"]
    log.info("rows per split: %s", meta["rows"])
    log.info("attack share per split: %s", meta["attack_share"])
    log.info("max class-share deviation: %.4f pts", checks["max_class_share_deviation"] * 100)
    log.info("feature-group overlap across splits: %s", checks["feature_group_overlap"])


if __name__ == "__main__":
    main()
