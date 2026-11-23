"""Build and validate the homogeneous host graph for the GNN stage.

Usage::

    uv run python scripts/build_graph.py
    uv run python scripts/build_graph.py --graph.node_features degree
    uv run python scripts/build_graph.py --graph.protocol unseen_host --graph.holdout 192.168.1.30

Checks the splits against the frozen baseline manifest first (verify_splits).
Exits with status 1 if a hard check fails.

Output:
- data/processed/graph/<protocol>[__<holdout>].pt   (torch_geometric Data)
- results/graph/<same name>/report.json, host_graph.png, degree_distributions.png
"""

import logging
import sys

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.protocols import attacker_hosts
from cyberdefense.data.ton_iot import NORMAL_CLASS, SPLITS, TARGET_MULTI, load_split
from cyberdefense.freeze import verify_splits
from cyberdefense.graph.build import build_graph, save_graph
from cyberdefense.graph.validation import (
    describe_graph,
    validate_graph,
    write_figures,
    write_report,
)
from cyberdefense.utils import setup_logging

log = logging.getLogger("build_graph")


def main() -> int:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    g = cfg.graph
    verify_splits(cfg, g.manifest)
    log.info("splits match the frozen manifest %s", g.manifest)
    splits = {s: load_split(cfg.ton_iot, s) for s in SPLITS}

    data = build_graph(splits, cfg)
    name = g.protocol + (f"__{g.holdout}" if g.holdout else "")
    log.info("%s", data)
    report = {
        "config": g.model_dump(mode="json"),
        "frozen_manifest_verified": str(g.manifest),
        "validation": validate_graph(data, splits),
        "description": describe_graph(data),
    }

    if g.validate_holdouts:
        classes = sorted(splits["train"][TARGET_MULTI].astype(str).unique())
        holdouts = [("unseen_attack", c) for c in classes if c != NORMAL_CLASS]
        hosts = attacker_hosts(splits, cfg.baseline.unseen_host_min_attack_flows)
        holdouts += [("unseen_host", h) for h in hosts]
        report["holdout_validation"] = {}
        for protocol, holdout in holdouts:
            variant = cfg.model_copy(
                update={"graph": g.model_copy(update={"protocol": protocol, "holdout": holdout})}
            )
            result = validate_graph(build_graph(splits, variant), splits)
            report["holdout_validation"][f"{protocol}:{holdout}"] = result["hard_failures"]
            log.info(
                "  %-13s %-15s %s",
                protocol,
                holdout,
                "ok" if result["passed"] else f"FAILED {result['hard_failures']}",
            )

    save_graph(data, g.out_dir / f"{name}.pt")
    out = g.report_dir / name
    write_report(report, out / "report.json")
    if g.make_figures:
        write_figures(data, out)

    d = report["description"]
    log.info(
        "graph: %d hosts, %s flow edges (%d edge features), %d distinct host pairs, "
        "%d self-loop flows, components %s",
        d["nodes"],
        f"{d['edges']:,}",
        d["edge_features"],
        d["distinct_directed_pairs"],
        d["self_loop_edges"],
        d["connected_components"],
    )
    for s, row in d["splits"].items():
        log.info(
            "  %-5s %s edges, attack share %.3f, %d hosts (%d without training edges)",
            s,
            f"{row['edges']:,}",
            row["attack_share"],
            row["hosts"],
            row["hosts_without_training_edges"],
        )
    failed = report["validation"]["hard_failures"] + [
        k for k, v in report.get("holdout_validation", {}).items() if v
    ]
    if failed:
        for f in failed:
            log.error("FAILED: %s", f)
        return 1
    log.info(
        "PASSED (%d checks%s). Graph: %s/%s.pt  Report: %s",
        len(report["validation"]["checks"]),
        f", {len(report.get('holdout_validation', {}))} holdout graphs"
        if g.validate_holdouts
        else "",
        g.out_dir,
        name,
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
