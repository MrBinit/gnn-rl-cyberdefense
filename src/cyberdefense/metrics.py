"""Evaluation metrics shared by every model, so comparisons are like for like.

Binary convention: 1 = attack (positive class), 0 = normal.

The data is imbalanced (78% attack; mitm is 0.5% of rows), so plain accuracy is
reported but never used to compare models. The headline metrics are:

- binary: F1, precision, recall (detection rate), FPR, MCC, balanced accuracy,
  and the threshold-free PR-AUC / ROC-AUC when scores are available;
- multiclass: macro-F1 (every class counts equally), balanced accuracy, MCC,
  and macro one-vs-rest ROC-AUC when probabilities are available.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def binary_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float] | None = None
) -> dict[str, float]:
    """Detection metrics for 0/1 labels; ``y_score`` is P(attack) for the AUCs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: float(v) for k, v in out.items()}


def multiclass_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Macro-averaged metrics for integer class labels ``0..n_classes-1``.

    ``y_proba`` has one column per class (in label order) for the macro ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and len(np.unique(y_true)) == y_proba.shape[1]:
        out["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {k: float(v) for k, v in out.items()}


def fpr_key(fpr: float) -> str:
    """Metric-name suffix for a target FPR: 0.0025 -> 'fpr0p25', 0.01 -> 'fpr1' (percent)."""
    return "fpr" + f"{fpr * 100:g}".replace(".", "p")


def threshold_at_fpr(normal_scores: Sequence[float], fpr: float) -> float:
    """Score threshold at which at most ``fpr`` of these normal flows score *above* it.

    Predict attack where ``score > threshold``. Ties (common with tree ensembles) can
    only lower the achieved FPR, never raise it above the target.
    """
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    scores = np.sort(np.asarray(normal_scores, dtype=float))[::-1]
    if scores.size == 0:
        raise ValueError("need at least one normal score to set a threshold")
    return float(scores[int(np.floor(fpr * scores.size))])


def fixed_fpr_metrics(
    *,
    y_val: Sequence[int],
    val_scores: Sequence[float],
    y_test: Sequence[int],
    test_scores: Sequence[float],
    fprs: Sequence[float],
    test_holdout: Sequence[bool] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Recall at fixed false-positive rates, with thresholds chosen on validation only.

    For each target FPR, the threshold is set on the *validation* normal flows, then
    applied unchanged to test. Comparing models at the same FPR separates real
    detection ability from a more permissive operating point.

    Returns ``(thresholds, val_metrics, test_metrics)`` keyed by ``fpr_key``:
    ``recall_at_<key>`` and the achieved ``fpr_at_<key>``; with ``test_holdout`` (a
    mask over test rows) also ``holdout_recall_at_<key>`` for its attack rows and
    ``holdout_fpr_at_<key>`` for its normal rows (if any).
    """
    y_val, val_scores = np.asarray(y_val), np.asarray(val_scores, dtype=float)
    y_test, test_scores = np.asarray(y_test), np.asarray(test_scores, dtype=float)
    thresholds, val_out, test_out = {}, {}, {}
    for fpr in fprs:
        key = fpr_key(fpr)
        t = threshold_at_fpr(val_scores[y_val == 0], fpr)
        thresholds[key] = t
        for y, s, out in ((y_val, val_scores, val_out), (y_test, test_scores, test_out)):
            flagged = s > t
            out[f"recall_at_{key}"] = float(flagged[y == 1].mean()) if (y == 1).any() else 0.0
            out[f"fpr_at_{key}"] = float(flagged[y == 0].mean()) if (y == 0).any() else 0.0
        if test_holdout is not None:
            held = np.asarray(test_holdout, dtype=bool)
            rows = held & (y_test == 1)
            if rows.any():
                test_out[f"holdout_recall_at_{key}"] = float((test_scores[rows] > t).mean())
            normal = held & (y_test == 0)
            if normal.any():  # unseen_host / unseen_pair: the holdout's own normal flows
                test_out[f"holdout_fpr_at_{key}"] = float((test_scores[normal] > t).mean())
    return thresholds, val_out, test_out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sound for small n and rates near 0 or 1)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # The interval always contains p; clamp so float rounding can't push it outside.
    return float(max(0.0, min(p, centre - half))), float(min(1.0, max(p, centre + half)))


def matched_fpr_metrics(
    *,
    y_test: Sequence[int],
    test_scores: Sequence[float],
    fprs: Sequence[float],
    test_holdout: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Recall at a **matched actual test FPR**: the threshold is set on the test normal flows.

    ``fixed_fpr_metrics`` picks thresholds on val (deployable), but when hosts are held out
    the val threshold lands at a different actual test FPR for every model, so recalls
    aren't comparable. Here every model is read off its own test ROC curve at the same
    actual FPR (an evaluation-only operating point; nothing is fitted). Ties can only
    lower the achieved FPR, which is reported as ``matched_fpr_at_<key>``.

    Keys: ``tpr_at_<key>`` (all test attacks), ``matched_fpr_at_<key>``, and with
    ``test_holdout`` (a mask over test rows) ``holdout_tpr_at_<key>`` for its attack rows
    with a 95% Wilson interval ``holdout_tpr_lo_at_<key>`` / ``holdout_tpr_hi_at_<key>``.
    """
    y, s = np.asarray(y_test), np.asarray(test_scores, dtype=float)
    out: dict[str, float] = {}
    for fpr in fprs:
        key = fpr_key(fpr)
        t = threshold_at_fpr(s[y == 0], fpr)
        flagged = s > t
        out[f"tpr_at_{key}"] = float(flagged[y == 1].mean()) if (y == 1).any() else 0.0
        out[f"matched_fpr_at_{key}"] = float(flagged[y == 0].mean())
        if test_holdout is not None:
            rows = np.asarray(test_holdout, dtype=bool) & (y == 1)
            if rows.any():
                hits, n = int(flagged[rows].sum()), int(rows.sum())
                lo, hi = wilson_interval(hits, n)
                out[f"holdout_tpr_at_{key}"] = hits / n
                out[f"holdout_tpr_lo_at_{key}"] = lo
                out[f"holdout_tpr_hi_at_{key}"] = hi
    return out


def holdout_metrics(
    y_pred: Sequence[int],
    attack_scores: Sequence[float],
    in_holdout: Sequence[bool],
    is_attack: Sequence[bool],
) -> tuple[dict[str, float], int]:
    """Detection of the held-out test flows (unseen_attack / unseen_host protocols).

    Returns ``(metrics, n_held_out_attack_flows)`` with ``holdout_recall`` (at the model's
    own threshold), the threshold-free ``holdout_roc_auc`` (held-out attacks ranked against
    all normal test flows), ``seen_attack_recall`` and, when the holdout has normal flows
    (unseen_host: the device's own traffic), ``holdout_normal_fpr``.
    """
    pred = np.asarray(y_pred)
    scores = np.asarray(attack_scores, dtype=float)
    held, attack = np.asarray(in_holdout, dtype=bool), np.asarray(is_attack, dtype=bool)
    target = held & attack
    out = {"holdout_recall": float(pred[target].mean()) if target.any() else 0.0}
    if target.any() and (~attack).any():
        ranked = target | ~attack
        out["holdout_roc_auc"] = float(roc_auc_score(attack[ranked], scores[ranked]))
    out["seen_attack_recall"] = (
        float(pred[~held & attack].mean()) if (~held & attack).any() else 0.0
    )
    if (held & ~attack).any():
        out["holdout_normal_fpr"] = float(pred[held & ~attack].mean())
    return out, int(target.sum())


def f1_threshold(
    y_true: Sequence[int], scores: Sequence[float], n_thresholds: int = 100, n_folds: int = 5
) -> float:
    """Decision threshold (predict attack where ``score >= t``) that maximises F1.

    Reproduces the baselines' ``TunedThresholdClassifierCV(FrozenEstimator(model),
    scoring="f1")`` exactly, so every model picks its threshold the same way: F1 curves
    on ``n_thresholds`` evenly spaced scores per stratified fold (unshuffled), linearly
    interpolated onto a common grid, averaged over folds, then the argmax. Validation only.
    """
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    curves = []
    for _, fold in StratifiedKFold(n_folds).split(s.reshape(-1, 1), y):
        grid = np.linspace(s[fold].min(), s[fold].max(), n_thresholds)
        f1s = [f1_score(y[fold], (s[fold] >= t).astype(int), zero_division=0) for t in grid]
        curves.append((grid, np.array(f1s)))
    lo, hi = min(g.min() for g, _ in curves), max(g.max() for g, _ in curves)
    if np.isclose(lo, hi):
        raise ValueError("constant scores: no threshold to tune")
    grid = np.linspace(lo, hi, n_thresholds)
    mean_f1 = np.mean([np.interp(grid, g, f) for g, f in curves], axis=0)
    return float(grid[int(np.argmax(mean_f1))])


def detection_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Pick binary or multiclass metrics from the labels (no scores)."""
    labels = set(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])).tolist())
    if labels <= {0, 1}:
        return binary_metrics(y_true, y_pred)
    return multiclass_metrics(y_true, y_pred)


def per_class_report(y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]) -> dict:
    """Precision / recall / F1 / support for every class, keyed by class name."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {name: report[name] for name in class_names}


def confusion(y_true: Sequence, y_pred: Sequence, n_classes: int) -> list[list[int]]:
    """Confusion matrix (rows = true, columns = predicted) as nested lists for JSON."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
