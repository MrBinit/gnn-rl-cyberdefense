"""Run every GNN model x task x protocol x holdout x seed, evaluated like the baselines.

For each (protocol, holdout) the host graph is built once with ``build_graph`` and
**validated** (``validate_graph``: held-out edges absent from train/val/message passing,
held-out host isolated, preprocessor fit on train edges, ...); a failing graph stops the
run. All models and seeds then train on that same graph.

Evaluation reuses the baselines' code: the F1 threshold is tuned on val with the same
search (``metrics.f1_threshold``), recall at fixed FPRs uses val-chosen thresholds
(``metrics.fixed_fpr_metrics``), held-out detection uses ``metrics.holdout_metrics``, and
the per-run JSON has the same keys, so ``cyberdefense.reporting`` tables mix both stages.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric.data import Data

from cyberdefense.config import ExperimentConfig, GNNModel, Protocol, Task
from cyberdefense.data.protocols import (
    HOLDOUT_PROTOCOLS,
    attack_host_holdouts_for,
    attacker_hosts,
    pair_holdouts,
    target_mask,
)
from cyberdefense.data.schema import NORMAL_CLASS, SPLITS, TARGET_BINARY, TARGET_MULTI
from cyberdefense.gnn.models import EdgeClassifier
from cyberdefense.gnn.shortcut import host_prior_proba
from cyberdefense.gnn.train import (
    GraphInputs,
    fit,
    predict_proba,
    prepare_inputs,
    without_self_loops,
)
from cyberdefense.graph.build import build_graph
from cyberdefense.graph.validation import validate_graph
from cyberdefense.metrics import (
    binary_metrics,
    confusion,
    f1_threshold,
    fixed_fpr_metrics,
    holdout_metrics,
    matched_fpr_metrics,
    multiclass_metrics,
    per_class_report,
)
from cyberdefense.reporting import GROUP_KEYS, aggregate, confusion_figure, summarize
from cyberdefense.tracking import log_metrics, make_writer
from cyberdefense.utils import seed_everything

log = logging.getLogger(__name__)

IMBALANCE = "class_weight"  # the only strategy for GNNs (resampling cannot touch a graph)


def experiment_name(cfg: ExperimentConfig) -> str:
    """gnn, plus non-default graph options and the ablation: gnn_transductive, gnn_no_self_loops.

    Window features as edge inputs add ``win`` (gnn_win).
    """
    parts = ["gnn"]
    if cfg.dataset.use_window_features:
        parts.append("win")
    if cfg.graph.message_passing != "train_only":
        parts.append(cfg.graph.message_passing)
    if cfg.graph.node_features != "constant":
        parts.append(cfg.graph.node_features)
    if cfg.gnn.ablation != "none":
        parts.append(cfg.gnn.ablation)
    return "_".join(parts)


@dataclass(frozen=True)
class GNNRunSpec:
    model: GNNModel
    task: Task
    protocol: Protocol
    seed: int
    holdout: str | None = None
    imbalance: str = IMBALANCE
    tuned_for: bool = False  # GNNs are untuned in this phase

    @property
    def run_name(self) -> str:
        suffix = f"__holdout_{self.holdout}" if self.holdout else ""
        return (
            f"{self.protocol}__{self.task}__{self.model}__{self.imbalance}__seed{self.seed}{suffix}"
        )


def iter_graphs(
    cfg: ExperimentConfig,
    attack_types: list[str],
    hosts: list[str],
    attack_host: list[str] | None = None,
) -> list[tuple[Protocol, str | None]]:
    """(protocol, holdout) pairs to build a graph for, filtered by ``gnn.holdouts``.

    ``gnn.holdouts`` matches a holdout's full name or, for unseen_attack_host, its type.
    """
    holdouts_for = {
        "random": [None],
        "unseen_attack": attack_types,
        "unseen_host": hosts,
        "unseen_pair": pair_holdouts(cfg.baseline.unseen_pair_folds),
        "unseen_attack_host": attack_host or [],
    }
    keep = cfg.gnn.holdouts
    return [
        (protocol, h)
        for protocol in cfg.gnn.protocols
        for h in holdouts_for[protocol]
        if h is None or keep is None or h in keep or h.partition("@")[0] in keep
    ]


GRAPH_MODELS = ("gcn", "sage", "gat")


def ablation_models(cfg: ExperimentConfig) -> list[GNNModel]:
    """The configured models an ablation actually changes (the others would repeat Phase 1).

    Structure ablations only change graph models; without propagation layers GCN, SAGE and
    GAT are the same network, so only the first listed one runs and stands for all three.
    """
    models, ablation = list(cfg.gnn.models), cfg.gnn.ablation
    if ablation in ("shuffled_structure", "topology_only", "no_propagation"):
        models = [m for m in models if m in GRAPH_MODELS]
    if ablation == "no_propagation":
        models = models[:1]
    return models


def iter_specs(cfg: ExperimentConfig, protocol: Protocol, holdout: str | None) -> list[GNNRunSpec]:
    g = cfg.gnn
    tasks = [t for t in g.tasks if protocol not in HOLDOUT_PROTOCOLS or t == "binary"]
    return [
        GNNRunSpec(model, task, protocol, seed, holdout)
        for task in tasks
        for model in ablation_models(cfg)
        for seed in g.seeds
    ]


def graph_for(
    cfg: ExperimentConfig, splits: dict[str, pd.DataFrame], protocol: Protocol, holdout: str | None
) -> Data:
    """Build and validate the graph for one protocol/holdout; raises if a check fails."""
    variant = cfg.model_copy(
        update={"graph": cfg.graph.model_copy(update={"protocol": protocol, "holdout": holdout})}
    )
    data = build_graph(splits, variant)
    report = validate_graph(data, splits)
    if not report["passed"]:
        raise RuntimeError(
            f"graph {protocol}:{holdout} failed validation: {report['hard_failures']}"
        )
    return data


def run_one(
    cfg: ExperimentConfig,
    spec: GNNRunSpec,
    data: Data,
    inputs: GraphInputs,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    seed_everything(spec.seed)
    g = cfg.gnn
    binary = spec.task == "binary"
    y = data.y if binary else data.y_multi
    labels = ["normal", "attack"] if binary else list(data.class_names)
    n_out = len(labels)
    idx = {name: getattr(data, f"{name}_mask").nonzero().squeeze(1).to(g.device) for name in SPLITS}

    writer = make_writer(
        cfg.run_tb_dir,
        experiment_name(cfg),
        spec.run_name,
        json.dumps({"gnn": g.model_dump(mode="json"), "graph": cfg.graph.model_dump(mode="json")}),
    )
    y_np = {name: y[i.cpu()].numpy() for name, i in idx.items()}
    scores, predict_us, fitted, n_params = {}, None, None, 0
    if spec.model == "host_prior":
        start = time.perf_counter()
        proba = host_prior_proba(data.edge_index, y, data.train_mask, data.num_nodes, n_out)
        fit_seconds = time.perf_counter() - start
        scores = {name: proba[i.cpu()].numpy() for name, i in idx.items()}
    else:
        model = EdgeClassifier(
            spec.model,
            inputs.node_in.shape[1],
            inputs.edge_attr.shape[1],
            n_out,
            g,
            use_edge_features=g.ablation != "topology_only",
            num_layers=0 if g.ablation == "no_propagation" else None,
        ).to(g.device)
        n_params = sum(p.numel() for p in model.parameters())
        fitted = fit(model, inputs, y, idx["train"], idx["val"], n_out, g, spec.seed, writer)
        fit_seconds = fitted.fit_seconds
        for name in SPLITS:
            start = time.perf_counter()
            scores[name] = predict_proba(model, inputs, idx[name]).numpy()
            if name == "test":
                predict_us = (time.perf_counter() - start) / len(idx[name]) * 1e6

    threshold = None
    if binary and cfg.baseline.tune_threshold:
        threshold = f1_threshold(y_np["val"], scores["val"][:, 1])

    def predict(s: np.ndarray) -> np.ndarray:
        if threshold is not None:
            return (s[:, 1] >= threshold).astype(np.int64)
        return s.argmax(1)

    pred = {name: predict(scores[name]) for name in SPLITS}
    metrics = {
        name: binary_metrics(y_np[name], pred[name], scores[name][:, 1])
        if binary
        else multiclass_metrics(y_np[name], pred[name], scores[name])
        for name in SPLITS
    }

    in_holdout = target_mask(test_df, spec.protocol, spec.holdout) if spec.holdout else None
    fixed_fpr_thresholds = None
    if binary and cfg.baseline.fixed_fprs:
        fixed_fpr_thresholds, val_extra, test_extra = fixed_fpr_metrics(
            y_val=y_np["val"],
            val_scores=scores["val"][:, 1],
            y_test=y_np["test"],
            test_scores=scores["test"][:, 1],
            fprs=cfg.baseline.fixed_fprs,
            test_holdout=in_holdout,
        )
        metrics["val"] |= val_extra
        metrics["test"] |= test_extra
        metrics["test"] |= matched_fpr_metrics(
            y_test=y_np["test"],
            test_scores=scores["test"][:, 1],
            fprs=cfg.baseline.fixed_fprs,
            test_holdout=in_holdout,
        )
    if binary:
        test_ei = data.edge_index[:, idx["test"].cpu()]
        metrics["test"] |= self_loop_fprs(
            y_np["test"],
            scores["test"][:, 1],
            pred["test"],
            (test_ei[0] == test_ei[1]).numpy(),
            fixed_fpr_thresholds or {},
        )

    result: dict = {
        "run": spec.run_name,
        **{k: getattr(spec, k) for k in (*GROUP_KEYS, "seed")},
        "rows": {name: len(i) for name, i in idx.items()},
        "threshold": threshold,
        "fixed_fpr_thresholds": fixed_fpr_thresholds,
        "best_epoch": fitted.best_epoch if fitted else None,
        "epochs_run": fitted.epochs_run if fitted else 0,
        "best_val_loss": fitted.best_val_loss if fitted else None,
        "fit_seconds": round(fit_seconds, 2),
        "predict_us_per_flow": round(predict_us, 3) if predict_us is not None else None,
        "parameters": n_params,
        "ablation": g.ablation,
        "graph": {
            "message_passing": data.message_passing,
            "node_features": cfg.graph.node_features,
            "message_passing_edges": int(data.mp_mask.sum()),
            "structure_edges": int(inputs.structure.shape[1]),
        },
        **metrics,
        "test_confusion": confusion(y_np["test"], pred["test"], n_out),
        "labels": labels,
        "history": fitted.history if fitted else [],
    }
    if not binary:
        result["test_per_class"] = per_class_report(y_np["test"], pred["test"], labels)
    if spec.holdout:
        is_attack = (test_df[TARGET_BINARY] == 1).to_numpy()
        extra, result["holdout_test_attack_flows"] = holdout_metrics(
            pred["test"], scores["test"][:, 1], in_holdout, is_attack
        )
        result["test"] |= extra

    run_dir = out_dir / spec.protocol / spec.task
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{spec.run_name}.json").write_text(json.dumps(result, indent=2))

    step = fitted.best_epoch if fitted else 0
    for name in SPLITS:
        log_metrics(writer, name, result[name], step)
    fig = confusion_figure(result["test_confusion"], labels, spec.run_name)
    if not spec.holdout:
        fig.savefig(run_dir / f"{spec.run_name}__confusion.png", dpi=120)
    writer.add_figure("confusion/test", fig, step)
    writer.add_hparams(
        {k: str(getattr(spec, k)) for k in (*GROUP_KEYS, "seed")},
        {f"hparam/test_{k}": v for k, v in result["test"].items()},
        run_name=".",
    )
    writer.close()
    return result


def self_loop_fprs(
    y: np.ndarray,
    attack_scores: np.ndarray,
    pred: np.ndarray,
    is_self_loop: np.ndarray,
    fixed_thresholds: dict[str, float],
) -> dict[str, float]:
    """Test FPR on self-loop vs other normal flows (every self-loop flow is normal).

    At the model's own threshold (``self_loop_fpr`` / ``non_self_loop_fpr``) and at each
    val-chosen fixed-FPR threshold (``..._fpr_at_<key>``, flagged where ``score > t``).
    """
    out = {}
    for name, rows in (("self_loop", is_self_loop), ("non_self_loop", ~is_self_loop)):
        normal = rows & (y == 0)
        if not normal.any():
            continue
        out[f"{name}_fpr"] = float(pred[normal].mean())
        for key, t in fixed_thresholds.items():
            out[f"{name}_fpr_at_{key}"] = float((attack_scores[normal] > t).mean())
    return out


def format_run(result: dict) -> str:
    """Short console report for one run."""
    test, val = result["test"], result["val"]
    if result["best_epoch"] is None:
        lines = [f"    {result['model']}: no training ({result['fit_seconds']:.2f}s)"]
    else:
        lines = [
            f"    {result['fit_seconds']:.1f}s, best epoch {result['best_epoch']}/"
            f"{result['epochs_run']}, val loss {result['best_val_loss']:.4f}"
        ]
    if result["task"] == "binary":
        lines.append(
            f"    test f1={test['f1']:.4f} recall={test['recall']:.4f} fpr={test['fpr']:.4f} "
            f"pr_auc={test.get('pr_auc', float('nan')):.4f} (val f1 {val['f1']:.4f})"
        )
        prefix = "holdout_recall_at_" if result["holdout"] else "recall_at_"
        fixed = [
            f"{k.removeprefix(prefix).removeprefix('fpr').replace('p', '.')}% FPR: {v:.4f}"
            for k, v in test.items()
            if k.startswith(prefix)
        ]
        if fixed:
            lines.append(
                "    "
                + ("held-out " if result["holdout"] else "")
                + "recall at "
                + ", ".join(fixed)
            )
        if result["holdout"]:
            lines.append(
                f"    held-out '{result['holdout']}': recall={test['holdout_recall']:.4f} on "
                f"{result['holdout_test_attack_flows']:,} flows, "
                f"roc_auc={test.get('holdout_roc_auc', float('nan')):.4f}"
            )
    else:
        lines.append(
            f"    test macro_f1={test['f1']:.4f} balanced_acc={test['balanced_accuracy']:.4f} "
            f"(val {val['f1']:.4f})"
        )
    return "\n".join(lines)


def run_all(cfg: ExperimentConfig, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every configured combination; write per-run JSON and the summary CSVs."""
    if missing := set(SPLITS) - set(splits):
        raise ValueError(f"missing splits: {sorted(missing)}")
    classes = sorted(splits["train"][TARGET_MULTI].astype(str).unique())
    attack_types = [c for c in classes if c != NORMAL_CLASS]
    hosts = attacker_hosts(splits, cfg.baseline.unseen_host_min_attack_flows)
    attack_host = (
        attack_host_holdouts_for(cfg, splits, attack_types)
        if "unseen_attack_host" in cfg.gnn.protocols
        else []
    )
    out_dir = cfg.run_dir / experiment_name(cfg)

    graphs = iter_graphs(cfg, attack_types, hosts, attack_host)
    total = sum(len(iter_specs(cfg, p, h)) for p, h in graphs)
    if not total:
        log.warning("no runs match the configuration")
        return pd.DataFrame()
    results, done = [], 0
    for protocol, holdout in graphs:
        specs = iter_specs(cfg, protocol, holdout)
        pending = [s for s in specs if not (cfg.gnn.resume and _result_path(out_dir, s).exists())]
        if len(pending) < len(specs):
            log.info(
                "%s:%s: reusing %d finished runs", protocol, holdout, len(specs) - len(pending)
            )
            results += [
                json.loads(_result_path(out_dir, s).read_text()) for s in specs if s not in pending
            ]
            done += len(specs) - len(pending)
        if not pending:
            continue
        data = graph_for(cfg, splits, protocol, holdout)
        log.info(
            "graph %s:%s validated (%d message-passing edges)",
            protocol,
            holdout,
            int(data.mp_mask.sum()),
        )
        if cfg.gnn.ablation == "no_self_loops":
            data = without_self_loops(data)  # after validation: only masks shrink
        inputs = None
        for spec in pending:
            done += 1
            log.info("[%d/%d] %s", done, total, spec.run_name)
            if inputs is None or cfg.gnn.ablation == "shuffled_structure":  # rewired per seed
                inputs = prepare_inputs(data, cfg.gnn.device, cfg.gnn.ablation, spec.seed)
            result = run_one(cfg, spec, data, inputs, splits["test"], out_dir)
            log.info("%s", format_run(result))
            results.append(result)

    summary = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    aggregate(summary).to_csv(out_dir / "summary_aggregated.csv", index=False)
    return summary


def _result_path(out_dir: Path, spec: GNNRunSpec) -> Path:
    return out_dir / spec.protocol / spec.task / f"{spec.run_name}.json"
