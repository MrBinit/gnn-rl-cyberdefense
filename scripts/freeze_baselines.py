"""Freeze the final baselines into frozen/baselines_<tag>/ (tracked in git).

Run after validation, the final (multi-seed) baseline runs and tuning. Usage::

    uv run python scripts/freeze_baselines.py
    uv run python scripts/freeze_baselines.py --freeze.tag v2
    uv run python scripts/freeze_baselines.py --freeze.overwrite

The snapshot holds the raw-data hash, split fingerprints, full config, environment,
git commit, validation report, tuned parameters and every baseline summary.
"""

import json
import logging

from cyberdefense.config import ExperimentConfig
from cyberdefense.freeze import freeze
from cyberdefense.utils import setup_logging

log = logging.getLogger("freeze_baselines")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    dest = freeze(cfg)
    manifest = json.loads((dest / "manifest.json").read_text())
    log.info("frozen -> %s", dest)
    log.info("splits: %s", {k: v["rows"] for k, v in manifest["splits"].items()})
    log.info("seeds per experiment: %s", manifest["seeds_per_experiment"])
    for rel in manifest["files"]:
        log.info("  %s", rel)
    if manifest["git"]["uncommitted_changes"]:
        log.warning("the working tree has uncommitted changes; commit them for a clean record")


if __name__ == "__main__":
    main()
