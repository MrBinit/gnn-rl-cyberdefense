"""Phase 2 leakage audit: hard checks on every GNN graph plus shortcut-exposure numbers.

No model results are needed and it takes a few minutes on CPU (two tiny trainings
check that test labels cannot influence training). Usage::

    uv run python scripts/audit_gnn.py --paths.results_dir results/phase2

Covers random, every unseen_attack type, every unseen_host host, every unseen_pair fold
and every feasible unseen_attack_host split (Phase 3). Writes <results_dir>/audit/
report.json, exposure.csv and attack_host_feasibility.csv; exits 1 if a check fails.
"""

import json
import logging
import sys

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.protocols import (
    attack_host_feasibility,
    attack_host_holdouts_for,
    attacker_hosts,
    pair_holdouts,
)
from cyberdefense.data.ton_iot import NORMAL_CLASS, SPLITS, TARGET_MULTI, load_split
from cyberdefense.freeze import verify_splits
from cyberdefense.gnn.audit import audit_graph, training_ignores_test_labels
from cyberdefense.gnn.experiment import graph_for
from cyberdefense.utils import setup_logging

log = logging.getLogger("audit_gnn")


def main() -> int:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    verify_splits(cfg, cfg.graph.manifest)
    splits = {s: load_split(cfg.ton_iot, s) for s in SPLITS}
    classes = sorted(splits["train"][TARGET_MULTI].astype(str).unique())
    graphs = [("random", None)]
    graphs += [("unseen_attack", c) for c in classes if c != NORMAL_CLASS]
    hosts = attacker_hosts(splits, cfg.baseline.unseen_host_min_attack_flows)
    graphs += [("unseen_host", h) for h in hosts]
    graphs += [("unseen_pair", p) for p in pair_holdouts(cfg.baseline.unseen_pair_folds)]
    attack_types = [c for c in classes if c != NORMAL_CLASS]
    feasibility = pd.DataFrame([attack_host_feasibility(splits, t) for t in attack_types])
    feasible = attack_host_holdouts_for(cfg, splits, attack_types)
    feasibility["feasible"] = feasibility["holdout"].isin(feasible)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        log.info(
            "unseen_attack_host feasibility (type + every host sending it held out):\n%s",
            feasibility.drop(columns="holdout"),
        )
    graphs += [("unseen_attack_host", h) for h in feasible]

    report, rows, failures = {"graphs": {}}, [], []
    for protocol, holdout in graphs:
        data = graph_for(cfg, splits, protocol, holdout)
        name = f"{protocol}:{holdout or '-'}"
        result = audit_graph(data, splits, cfg)
        failed = [k for k, v in result["checks"].items() if not v["passed"]]
        failures += [f"{name} {k}" for k in failed]
        report["graphs"][name] = result
        rows.append({"protocol": protocol, "holdout": holdout or "-", **result["exposure"]})
        log.info("%-30s %s", name, "ok" if not failed else f"FAILED {failed}")
        if protocol == "random":
            log.info("training twice to check test labels/features cannot influence training")
            invariance = training_ignores_test_labels(data)
            report["training_ignores_test_labels"] = invariance
            log.info("  %s %s", "ok" if invariance["passed"] else "FAILED", invariance["detail"])
            if not invariance["passed"]:
                failures.append("training_ignores_test_labels")

    out = cfg.paths.results_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report["failures"] = failures
    (out / "report.json").write_text(json.dumps(report, indent=2))
    feasibility.to_csv(out / "attack_host_feasibility.csv", index=False)
    exposure = pd.DataFrame(rows)
    exposure.to_csv(out / "exposure.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        log.info("shortcut exposure (share of test / held-out target flows):\n%s", exposure)
    if failures:
        for f in failures:
            log.error("FAILED: %s", f)
        return 1
    log.info("PASSED: %d graphs, no leakage found. Report: %s", len(graphs), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
