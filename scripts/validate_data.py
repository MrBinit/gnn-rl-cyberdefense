"""Validate the NetFlow copy, the preparation and every split set (after prepare_data).

Usage::

    uv run python scripts/validate_data.py
    uv run python scripts/validate_data.py --validation.official_csv path/to/other_copy.csv

Provenance: the raw class counts are compared with the published NF-CICIDS2018-v3 file
(UQ NetFlow v3, docs/08). Exits with status 1 if a hard check fails.

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
        "provenance: %s raw rows vs %s in the published file -> %s",
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

    prep = report["preparation"]
    log.info(
        "preparation: reproducible=%s, %s features, forbidden=%s, non-finite=%s, "
        "noisy-flag mismatches=%s",
        prep["reproducible_from_raw"],
        prep["n_features"],
        prep["forbidden_feature_columns"],
        prep["non_finite_feature_values"],
        prep["noisy_flag_mismatches"],
    )
    log.info(
        "windows: %s kept (%s with clean attacks); incomplete normal traffic in %s; "
        "cells off the attack cap: %s; rows not matching complete traffic: %s",
        prep["windows_kept"],
        prep["windows_with_clean_attacks"],
        prep["windows_with_incomplete_normal_traffic"],
        prep["window_type_cells_off_attack_cap"],
        prep["rows_not_matching_complete_traffic"],
    )
    prepared = report["prepared_splits"]
    log.info("exact duplicates: %s", prepared["exact_duplicate_rows_after_cleaning"])
    for name, s in prepared["split_sets"].items():
        log.info(
            "%-24s rows=%s excluded=%s test attacks=%s; seen attackers %.0f%%; masked rows=%s",
            name,
            s["rows"],
            s["excluded_rows"],
            s["test_attack_types"],
            s["test_attack_flows_from_hosts_seen_attacking_in_train"] * 100,
            s["masked_host_rows_in_train_val"],
        )
    if report["passed"]:
        log.info("PASSED. Report: %s/report.json", cfg.validation.out_dir)
        return 0
    for failure in report["hard_failures"]:
        log.error("FAILED: %s", failure)
    return 1


if __name__ == "__main__":
    sys.exit(main())
