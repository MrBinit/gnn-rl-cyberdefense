import numpy as np
import pytest

from cyberdefense.metrics import (
    binary_metrics,
    confusion,
    detection_metrics,
    fixed_fpr_metrics,
    fpr_key,
    multiclass_metrics,
    per_class_report,
    threshold_at_fpr,
)


def test_binary_metrics() -> None:
    m = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["fpr"] == pytest.approx(0.5)
    assert m["fnr"] == pytest.approx(0.0)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert "roc_auc" not in m  # no scores given


def test_binary_metrics_with_scores() -> None:
    m = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == m["pr_auc"] == 1.0
    assert m["mcc"] == pytest.approx(1.0)


def test_imbalance_exposes_trivial_classifier() -> None:
    # 90% attack: "always attack" has high accuracy but is useless.
    y_true = np.array([1] * 90 + [0] * 10)
    m = binary_metrics(y_true, np.ones(100, dtype=int))
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["fpr"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["mcc"] == pytest.approx(0.0)


def test_multiclass_macro_counts_rare_class_equally() -> None:
    y_true = [0] * 98 + [1, 1]
    y_pred = [0] * 100  # misses the rare class completely
    m = multiclass_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.98)
    assert m["recall"] == pytest.approx(0.5)  # macro: (1.0 + 0.0) / 2
    assert m["weighted_f1"] > m["f1"]


def test_multiclass_roc_auc_with_proba() -> None:
    y_true = [0, 1, 2]
    proba = np.eye(3)
    m = multiclass_metrics(y_true, [0, 1, 2], proba)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_detection_metrics_dispatch() -> None:
    assert "fpr" in detection_metrics([0, 1], [0, 1])
    assert "weighted_f1" in detection_metrics([0, 1, 2], [0, 1, 2])


def test_per_class_report_includes_absent_classes() -> None:
    report = per_class_report([0, 1, 1], [0, 1, 0], ["normal", "ddos", "mitm"])
    assert set(report) == {"normal", "ddos", "mitm"}
    assert report["ddos"]["recall"] == pytest.approx(0.5)
    assert report["mitm"]["support"] == 0


def test_confusion_shape() -> None:
    assert confusion([0, 1], [1, 1], n_classes=3) == [[0, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_fpr_key() -> None:
    assert fpr_key(0.0025) == "fpr0p25"
    assert fpr_key(0.01) == "fpr1"


def test_threshold_at_fpr_caps_false_positives() -> None:
    normal = np.arange(1000) / 1000  # 0.000 .. 0.999
    t = threshold_at_fpr(normal, 0.01)
    assert (normal > t).mean() == pytest.approx(0.01)
    # Ties can only lower the achieved FPR, never exceed the target.
    tied = np.r_[np.zeros(990), np.ones(10)]
    assert (tied > threshold_at_fpr(tied, 0.0025)).mean() <= 0.0025
    with pytest.raises(ValueError, match="fpr"):
        threshold_at_fpr(normal, 0.0)
    with pytest.raises(ValueError, match="normal score"):
        threshold_at_fpr([], 0.01)


def test_fixed_fpr_metrics_threshold_from_val_only() -> None:
    rng = np.random.default_rng(0)
    y_val = np.r_[np.zeros(1000), np.ones(500)].astype(int)
    val_scores = np.r_[rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1, 500)]
    # Test normals score much higher: a test-chosen threshold would hide that.
    y_test = np.r_[np.zeros(100), np.ones(100)].astype(int)
    test_scores = np.r_[np.full(100, 0.99), np.full(100, 0.995)]
    holdout = np.r_[np.zeros(150, dtype=bool), np.ones(50, dtype=bool)]
    thresholds, val, test = fixed_fpr_metrics(
        y_val=y_val,
        val_scores=val_scores,
        y_test=y_test,
        test_scores=test_scores,
        fprs=[0.01],
        test_holdout=holdout,
    )
    assert thresholds["fpr1"] < 0.6
    assert val["fpr_at_fpr1"] <= 0.01
    assert test["fpr_at_fpr1"] == 1.0  # the val threshold is applied unchanged
    assert test["recall_at_fpr1"] == test["holdout_recall_at_fpr1"] == 1.0
