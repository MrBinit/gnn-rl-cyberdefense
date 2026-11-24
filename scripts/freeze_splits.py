"""Freeze the prepared data and every split set into frozen/data_<tag>/ (tracked in git).

Run after scripts/prepare_data.py and a passing scripts/validate_data.py. Usage::

    uv run python scripts/freeze_splits.py
    uv run python scripts/freeze_splits.py --freeze.overwrite

Later stages call ``verify_splits`` against frozen/data_<tag>/manifest.json, so every model
provably trains and tests on the same rows (choose the set with --dataset.split_set).
"""

import json
import logging

from cyberdefense.config import ExperimentConfig
from cyberdefense.freeze import freeze_data
from cyberdefense.utils import setup_logging

log = logging.getLogger("freeze_splits")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    dest = freeze_data(cfg)
    manifest = json.loads((dest / "manifest.json").read_text())
    log.info("frozen -> %s", dest)
    for name, splits in manifest["split_sets"].items():
        log.info("  %-24s %s", name, {k: v["rows"] for k, v in splits.items()})
    if manifest["git"]["uncommitted_changes"]:
        log.warning("the working tree has uncommitted changes; commit them for a clean record")


if __name__ == "__main__":
    main()
