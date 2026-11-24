"""Run traditional-ML baselines over every configured combination.

Protocols:
- ``random``: the stratified train/val/test split as is.
- ``unseen_attack``: for each attack type, remove it from train and val, then
  measure on the full test split how much of that type is still flagged as an
  attack (binary task only; a classifier cannot name a class it never saw).
- ``unseen_host``: for each attacker host, remove every flow it sends or receives
  from train and val, then measure how many of its test attack flows are still
  flagged: attacks from an unfamiliar device (binary task only).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier

from cyberdefense.baselines.models import FittedBaseline, fit_baseline
from cyberdefense.config import BaselineModel, ExperimentConfig, ImbalanceStrategy, Protocol, Task
from cyberdefense.data.protocols import (
    HOLDOUT_PROTOCOLS,
    attack_host_holdouts_for,
    attacker_hosts,
    holdout_mask,
    pair_holdouts,
    target_mask,
)
from cyberdefense.data.schema import NORMAL_CLASS, SPLITS, TARGET_BINARY, TARGET_MULTI
from cyberdefense.metrics import (
    binary_metrics,
    confusion,
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


def experiment_name(cfg: ExperimentConfig) -> str:
    """Untuned and each tuning variant get their own result and TensorBoard folders.

    tuning -> baselines_tuned; tuning_disjoint -> baselines_tuned_disjoint.
    """
    if not cfg.baseline.use_tuned:
        return "baselines"
    suffix = cfg.tuning.out_dir.name.removeprefix("tuning").strip("_")
    return f"baselines_tuned_{suffix}" if suffix else "baselines_tuned"


@dataclass(frozen=True)
class RunSpec:
    model: BaselineModel
    task: Task
    protocol: Protocol
    imbalance: ImbalanceStrategy
    seed: int
    holdout: str | None = None
    tuned_for: bool = False  # holdout was a tuning target, so its score is optimistic

    @property
    def run_name(self) -> str:
        suffix = f"__holdout_{self.holdout}" if self.holdout else ""
        return (
            f"{self.protocol}__{self.task}__{self.model}__{self.imbalance}__seed{self.seed}{suffix}"
        )


def iter_specs(
    cfg: ExperimentConfig,
    attack_types: list[str],
    hosts: list[str] | None = None,
    attack_host: list[str] | None = None,
) -> list[RunSpec]:
    b = cfg.baseline
    holdouts_for = {
        "random": [None],
        "unseen_attack": attack_types,
        "unseen_host": hosts or [],
        "unseen_pair": pair_holdouts(b.unseen_pair_folds),
        "unseen_attack_host": attack_host or [],
    }
    tuned = set(cfg.tuning.holdouts) if b.use_tuned else set()
    specs = []
    for protocol in b.protocols:
        for task in b.tasks:
            if protocol in HOLDOUT_PROTOCOLS and task != "binary":
                continue
            specs += [
                RunSpec(
                    model,
                    task,
                    protocol,
                    imbalance,
                    seed,
                    holdout,
                    tuned_for=protocol == "unseen_attack" and holdout in tuned,
                )
                for holdout in holdouts_for[protocol]
                for imbalance in b.imbalance
                for model in b.models
                for seed in b.seeds
            ]
    return specs


def encode_targets(df: pd.DataFrame, task: Task, classes: list[str]) -> np.ndarray:
    if task == "binary":
        return df[TARGET_BINARY].to_numpy(dtype=np.int64)
    codes = pd.Categorical(df[TARGET_MULTI], categories=classes).codes.astype(np.int64)
    if (codes < 0).any():
        raise ValueError("found an attack type that is not in the class list")
    return codes


def evaluate(
    fitted: FittedBaseline, df: pd.DataFrame, y_true: np.ndarray, task: Task
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Metrics, predictions and class probabilities for one split."""
    scores = fitted.predict_scores(df)
    y_pred = fitted.predict(df, scores)
    if task == "binary":
        return binary_metrics(y_true, y_pred, scores[:, 1]), y_pred, scores
    return multiclass_metrics(y_true, y_pred, scores), y_pred, scores


def _log_xgb_curves(writer: SummaryWriter, model: XGBClassifier) -> None:
    """Per-boosting-round train/val curves, the tree-model equivalent of epochs."""
    for split, history in zip(("train", "val"), model.evals_result().values(), strict=True):
        for metric, values in history.items():
            for step, value in enumerate(values, start=1):
                writer.add_scalar(f"{metric}/{split}", value, step)


def run_one(
    cfg: ExperimentConfig,
    spec: RunSpec,
    data: dict[str, pd.DataFrame],
    classes: list[str],
    out_dir: Path,
) -> dict:
    seed_everything(spec.seed)
    train, val, test = data["train"], data["val"], data["test"]
    if spec.holdout:
        train = train[~holdout_mask(train, spec.protocol, spec.holdout)]
        val = val[~holdout_mask(val, spec.protocol, spec.holdout)]
    splits = {"train": train, "val": val, "test": test}
    y = {name: encode_targets(df, spec.task, classes) for name, df in splits.items()}

    start = time.perf_counter()
    fitted = fit_baseline(
        model_name=spec.model,
        task=spec.task,
        imbalance=spec.imbalance,
        seed=spec.seed,
        cfg=cfg.baseline,
        data_cfg=cfg.dataset,
        train=train,
        y_train=y["train"],
        val=val,
        y_val=y["val"],
    )
    fit_seconds = time.perf_counter() - start

    metrics, predictions, scores = {}, {}, {}
    for name, df in splits.items():
        start = time.perf_counter()
        metrics[name], predictions[name], scores[name] = evaluate(fitted, df, y[name], spec.task)
        if name == "test":
            predict_us = (time.perf_counter() - start) / len(df) * 1e6

    in_holdout = target_mask(test, spec.protocol, spec.holdout) if spec.holdout else None
    fixed_fpr_thresholds = None
    if spec.task == "binary" and cfg.baseline.fixed_fprs:
        fixed_fpr_thresholds, val_extra, test_extra = fixed_fpr_metrics(
            y_val=y["val"],
            val_scores=scores["val"][:, 1],
            y_test=y["test"],
            test_scores=scores["test"][:, 1],
            fprs=cfg.baseline.fixed_fprs,
            test_holdout=in_holdout,
        )
        metrics["val"] |= val_extra
        metrics["test"] |= test_extra
        metrics["test"] |= matched_fpr_metrics(
            y_test=y["test"],
            test_scores=scores["test"][:, 1],
            fprs=cfg.baseline.fixed_fprs,
            test_holdout=in_holdout,
        )

    labels = ["normal", "attack"] if spec.task == "binary" else classes
    result: dict = {
        "run": spec.run_name,
        **{k: getattr(spec, k) for k in (*GROUP_KEYS, "seed")},
        "rows": {name: len(df) for name, df in splits.items()},
        "train_rows_after_resampling": fitted.train_rows_after_resampling,
        "threshold": fitted.threshold,
        "fixed_fpr_thresholds": fixed_fpr_thresholds,
        "best_iteration": fitted.best_iteration,
        "fit_seconds": round(fit_seconds, 2),
        "predict_us_per_flow": round(predict_us, 3),
        **metrics,
        "test_confusion": confusion(y["test"], predictions["test"], len(labels)),
        "labels": labels,
    }
    if spec.task == "multiclass":
        result["test_per_class"] = per_class_report(y["test"], predictions["test"], classes)
    if spec.holdout:
        is_attack = (test[TARGET_BINARY] == 1).to_numpy()
        extra, result["holdout_test_attack_flows"] = holdout_metrics(
            predictions["test"], scores["test"][:, 1], in_holdout, is_attack
        )
        result["test"] |= extra

    run_dir = out_dir / spec.protocol / spec.task
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{spec.run_name}.json").write_text(json.dumps(result, indent=2))

    writer = make_writer(
        cfg.run_tb_dir,
        experiment_name(cfg),
        spec.run_name,
        cfg.baseline.model_dump_json(indent=2),
    )
    model = fitted.pipeline.named_steps["model"]
    if isinstance(model, XGBClassifier):
        _log_xgb_curves(writer, model)
    step = fitted.best_iteration or 0
    for name in SPLITS:
        log_metrics(writer, name, result[name], step)
    fig = confusion_figure(result["test_confusion"], labels, spec.run_name)
    if not spec.holdout:  # one PNG per random-split run; holdout runs stay in JSON/TB
        fig.savefig(run_dir / f"{spec.run_name}__confusion.png", dpi=120)
    writer.add_figure("confusion/test", fig, step)  # closes the figure
    writer.add_hparams(
        {k: str(getattr(spec, k)) for k in (*GROUP_KEYS, "seed")},
        {f"hparam/test_{k}": v for k, v in result["test"].items()},
        run_name=".",
    )
    writer.close()
    return result


def format_run(result: dict) -> str:
    """Multi-line console report for one run: what to check while it trains."""
    train, val, test = result["train"], result["val"], result["test"]
    fit = f"fit {result['fit_seconds']:.1f}s"
    if result["best_iteration"] is not None:
        fit += f", best round {result['best_iteration']}"
    if result["train_rows_after_resampling"] != result["rows"]["train"]:
        fit += (
            f", train rows {result['rows']['train']:,} -> {result['train_rows_after_resampling']:,}"
        )
    lines = [f"    {fit}"]

    if result["task"] == "binary":
        threshold = f"{result['threshold']:.3f}" if result["threshold"] is not None else "0.5"
        lines.append(
            f"    test  f1={test['f1']:.4f} precision={test['precision']:.4f} "
            f"recall={test['recall']:.4f} fpr={test['fpr']:.4f} mcc={test['mcc']:.4f} "
            f"pr_auc={test.get('pr_auc', float('nan')):.4f} threshold={threshold}"
        )
        prefix = "holdout_recall_at_" if result["holdout"] else "recall_at_"
        fixed = [
            f"{k.removeprefix(prefix).removeprefix('fpr').replace('p', '.')}%: "
            f"{v:.4f} (test fpr {test['fpr_at_' + k.removeprefix(prefix)]:.4f})"
            for k, v in test.items()
            if k.startswith(prefix)
        ]
        if fixed:
            what = "held-out recall" if result["holdout"] else "recall"
            lines.append(f"    {what} at val-chosen FPR " + ", ".join(fixed))
        if result["holdout"]:
            kind = "host" if result["protocol"] == "unseen_host" else "attack"
            tag = " [tuned for]" if result["tuned_for"] else ""
            auc = test.get("holdout_roc_auc", float("nan"))
            lines.append(
                f"    unseen {kind} '{result['holdout']}'{tag}: "
                f"recall={test['holdout_recall']:.4f} on {result['holdout_test_attack_flows']:,} "
                f"flows (seen attacks {test['seen_attack_recall']:.4f}), roc_auc={auc:.4f}"
            )
    else:
        lines.append(
            f"    test  macro_f1={test['f1']:.4f} balanced_acc={test['balanced_accuracy']:.4f} "
            f"mcc={test['mcc']:.4f} roc_auc={test.get('roc_auc', float('nan')):.4f}"
        )
        per_class = result["test_per_class"]
        worst = sorted(per_class, key=lambda c: per_class[c]["f1-score"])[:3]
        lines.append(
            "    weakest classes: "
            + ", ".join(
                f"{c} f1={per_class[c]['f1-score']:.3f} "
                f"(p={per_class[c]['precision']:.2f} r={per_class[c]['recall']:.2f})"
                for c in worst
            )
        )

    gap = train["f1"] - test["f1"]
    flag = "  <- overfitting?" if gap > 0.05 else ""
    lines.append(
        f"    f1 train={train['f1']:.4f} val={val['f1']:.4f} test={test['f1']:.4f} "
        f"(train-test gap {gap:+.4f}){flag}"
    )
    return "\n".join(lines)


def run_all(cfg: ExperimentConfig, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every configured combination; write per-run JSON and the summary CSVs."""
    if missing := set(SPLITS) - set(data):
        raise ValueError(f"missing splits: {sorted(missing)}")
    classes = sorted(data["train"][TARGET_MULTI].astype(str).unique())
    attack_types = [c for c in classes if c != NORMAL_CLASS]
    hosts = attacker_hosts(data, cfg.baseline.unseen_host_min_attack_flows)
    attack_host = (
        attack_host_holdouts_for(cfg, data, attack_types)
        if "unseen_attack_host" in cfg.baseline.protocols
        else []
    )
    out_dir = cfg.run_dir / experiment_name(cfg)

    results = []
    specs = iter_specs(cfg, attack_types, hosts, attack_host)
    if not specs:
        log.warning("no runs match the configuration (e.g. no attacker host passes the threshold)")
        return pd.DataFrame()
    for i, spec in enumerate(specs, start=1):
        log.info("[%d/%d] %s", i, len(specs), spec.run_name)
        result = run_one(cfg, spec, data, classes, out_dir)
        log.info("%s", format_run(result))
        results.append(result)

    summary = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    aggregate(summary).to_csv(out_dir / "summary_aggregated.csv", index=False)
    return summary
