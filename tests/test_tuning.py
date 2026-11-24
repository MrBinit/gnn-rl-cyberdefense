import json

import numpy as np
import pandas as pd
import pytest

from cyberdefense.baselines.experiment import encode_targets, experiment_name
from cyberdefense.baselines.models import fit_baseline
from cyberdefense.baselines.tuning import (
    RF_DEFAULTS,
    XGB_DEFAULTS,
    apply_tuned,
    objective_value,
    prepare_tuning_data,
    tune,
)
from cyberdefense.config import (
    BaselineConfig,
    ExperimentConfig,
    ObjectiveWeights,
    RandomForestConfig,
    TuningConfig,
    XGBoostConfig,
)
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.schema import load_split


@pytest.fixture
def tune_cfg(tmp_cfg: ExperimentConfig) -> ExperimentConfig:
    cfg = tmp_cfg.model_copy(
        update={
            "baseline": BaselineConfig(
                random_forest=RandomForestConfig(n_estimators=10),
                xgboost=XGBoostConfig(n_estimators=20, early_stopping_rounds=5),
            ),
            "tuning": TuningConfig(
                rf_trials=3,
                xgb_trials=3,
                holdouts=["mitm", "ddos"],
                out_dir=tmp_cfg.paths.results_dir / "tuning",
            ),
            "dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False}),
        }
    )
    build_dataset(cfg)
    return cfg


@pytest.fixture
def train_val(tune_cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split(tune_cfg.dataset, "train"), load_split(tune_cfg.dataset, "val")


def test_objective_value() -> None:
    weights = ObjectiveWeights(macro_f1=0.5, unseen_recall=0.5, fpr=2.0)
    parts = {"macro_f1": 0.9, "unseen_recall": 0.8, "fpr": 0.01}
    assert objective_value(parts, weights) == pytest.approx(0.45 + 0.4 - 0.02)


def test_defaults_match_untuned_configs() -> None:
    assert XGB_DEFAULTS["learning_rate"] == XGBoostConfig().learning_rate
    assert XGB_DEFAULTS["class_weight_power"] == 1.0
    assert RF_DEFAULTS["n_estimators"] == RandomForestConfig().n_estimators


def test_unknown_holdout_rejected(tune_cfg: ExperimentConfig, train_val) -> None:
    cfg = tune_cfg.model_copy(update={"tuning": TuningConfig(holdouts=["not-an-attack"])})
    with pytest.raises(ValueError, match="holdouts"):
        prepare_tuning_data(cfg, *train_val)


def test_train_fraction_is_stratified(tune_cfg: ExperimentConfig, train_val) -> None:
    cfg = tune_cfg.model_copy(
        update={"tuning": tune_cfg.tuning.model_copy(update={"train_fraction": 0.5})}
    )
    data = prepare_tuning_data(cfg, *train_val)
    full = train_val[0]
    assert len(data.train) == pytest.approx(len(full) / 2, abs=2)
    share = data.train["attack_type"].value_counts(normalize=True)
    full_share = full["attack_type"].value_counts(normalize=True)
    assert (share - full_share).abs().max() < 0.02
    assert len(data.val) == len(train_val[1])  # val is never subsampled


def test_class_weight_power_zero_equals_unweighted(tune_cfg: ExperimentConfig, train_val) -> None:
    train, val = train_val
    classes = sorted(train["attack_type"].astype(str).unique())
    y_train = encode_targets(train, "multiclass", classes)
    y_val = encode_targets(val, "multiclass", classes)

    def fit(imbalance: str, power: float) -> np.ndarray:
        rf = tune_cfg.baseline.random_forest.model_copy(update={"class_weight_power": power})
        fitted = fit_baseline(
            model_name="random_forest",
            task="multiclass",
            imbalance=imbalance,
            seed=0,
            cfg=tune_cfg.baseline.model_copy(update={"random_forest": rf}),
            data_cfg=tune_cfg.dataset,
            train=train,
            y_train=y_train,
            val=val,
            y_val=y_val,
        )
        return fitted.predict_scores(val)

    np.testing.assert_allclose(fit("class_weight", 0.0), fit("none", 1.0))


def test_tune_end_to_end(tune_cfg: ExperimentConfig, train_val) -> None:
    results = tune(tune_cfg, *train_val)
    assert set(results) == {"random_forest", "xgboost"}

    for model_name, best in results.items():
        out = tune_cfg.tuning.out_dir / model_name
        trials = pd.read_csv(out / "trials.csv")
        assert trials["is_default"].sum() == 1
        assert trials.loc[trials["is_default"], "trial"].iloc[0] == 0
        for col in ("objective", "macro_f1", "unseen_recall", "fpr", "unseen_recall_mitm"):
            assert trials[col].notna().all()
        saved = json.loads((out / "best_params.json").read_text())
        assert saved["params"] == best["params"]
        assert saved["default_val_scores"] is not None
        assert saved["val_scores"]["objective"] == pytest.approx(trials["objective"].max())

    assert len(pd.read_csv(tune_cfg.tuning.out_dir / "random_forest" / "trials.csv")) == 3
    assert (tune_cfg.tuning.out_dir / "xgboost" / "study.db").exists()
    assert any((tune_cfg.paths.tb_dir / "tuning").iterdir())

    tuned = apply_tuned(
        tune_cfg.model_copy(
            update={"baseline": tune_cfg.baseline.model_copy(update={"use_tuned": True})}
        )
    )
    assert (
        tuned.baseline.random_forest.max_features
        == results["random_forest"]["params"]["max_features"]
    )
    assert tuned.baseline.xgboost.max_depth == results["xgboost"]["params"]["max_depth"]
    assert experiment_name(tuned) == "baselines_tuned"


def test_xgb_study_resumes_without_exceeding_budget(tune_cfg: ExperimentConfig, train_val) -> None:
    cfg = tune_cfg.model_copy(
        update={"tuning": tune_cfg.tuning.model_copy(update={"models": ["xgboost"]})}
    )
    tune(cfg, *train_val)
    first = pd.read_csv(cfg.tuning.out_dir / "xgboost" / "trials.csv")
    tune(cfg, *train_val)  # budget already used: no new trials
    second = pd.read_csv(cfg.tuning.out_dir / "xgboost" / "trials.csv")
    assert len(second) == len(first)


def test_apply_tuned_requires_tuning_results(tmp_cfg: ExperimentConfig) -> None:
    with pytest.raises(FileNotFoundError, match="tune_baselines"):
        apply_tuned(
            tmp_cfg.model_copy(
                update={"tuning": TuningConfig(out_dir=tmp_cfg.paths.results_dir / "none")}
            )
        )
