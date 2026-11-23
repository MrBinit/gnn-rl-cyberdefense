"""Validate the ToN-IoT copy and the prepared splits (run scripts/prepare_data.py first).

Usage::

    uv run python scripts/validate_data.py
    uv run python scripts/validate_data.py --validation.official_csv path/to/Train_Test_Network.csv

The official file comes from the UNSW ToN-IoT page (Train_Test_datasets folder);
it needs a manual download. Exits with status 1 if a hard check fails.

Output: results/validation/report.json
"""

import logging
import sys

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.validation import validate
from cyberdefense.utils import setup_logging

log = logging.getLogger("validate_data")


def main() -> int:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    report = validate(cfg)

    reported = report["reported_official_comparison"]
    log.info(
        "provenance: %s rows vs %s reported for the official file -> %s",
        f"{reported['this_copy_rows']:,}",
        f"{reported['reported_official_rows']:,}",
        reported["verdict"],
    )
    for cls in reported["differing_classes"]:
        row = reported["per_class"][cls]
        log.info("  %s: %s here vs %s reported", cls, row["this_copy"], row["reported_official"])
    if official := report.get("official_comparison"):
        log.info(
            "official file: %s rows; rows of this copy found in it: %.2f%% "
            "(attack %.2f%%, normal %.2f%%)",
            f"{official['official_rows']:,}",
            official.get("share_of_rows_found_in_official", float("nan")) * 100,
            official.get("share_of_attack_rows_found", float("nan")) * 100,
            official.get("share_of_normal_rows_found", float("nan")) * 100,
        )
    else:
        log.info("official file not supplied; pass --validation.official_csv to compare rows")

    splits = report["prepared_splits"]
    log.info("splits: %s", splits["rows"])
    log.info("row overlap between splits: %s", splits["row_overlap_between_splits"])
    log.info("feature-group overlap: %s", splits["feature_group_overlap_between_splits"])
    log.info("max class-share deviation: %.4f", splits["max_class_share_deviation"])
    log.info(
        "test attack flows from hosts already seen attacking in train: %.1f%% "
        "(why the unseen_host protocol is needed)",
        splits["test_attack_flows_from_hosts_seen_attacking_in_train"] * 100,
    )
    if report["passed"]:
        log.info("PASSED. Report: %s/report.json", cfg.validation.out_dir)
        return 0
    for failure in report["hard_failures"]:
        log.error("FAILED: %s", failure)
    return 1


if __name__ == "__main__":
    sys.exit(main())
