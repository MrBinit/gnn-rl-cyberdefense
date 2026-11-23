"""Estimators, imbalance handling and the fitted scikit-learn pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from cyberdefense.config import (
    BaselineConfig,
    BaselineModel,
    ImbalanceStrategy,
    Task,
    TonIoTConfig,
)
from cyberdefense.data.preprocess import make_preprocessor

Estimator = RandomForestClassifier | XGBClassifier
Sampler = RandomUnderSampler | RandomOverSampler | SMOTE


def make_estimator(name: BaselineModel, task: Task, cfg: BaselineConfig, seed: int) -> Estimator:
    if name == "random_forest":
        rf = cfg.random_forest
        return RandomForestClassifier(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_split=rf.min_samples_split,
            min_samples_leaf=rf.min_samples_leaf,
            max_features=rf.max_features,
            max_samples=rf.max_samples,
            criterion=rf.criterion,
            n_jobs=rf.n_jobs,
            random_state=seed,
        )
    if name == "xgboost":
        xgb = cfg.xgboost
        return XGBClassifier(
            n_estimators=xgb.n_estimators,
            learning_rate=xgb.learning_rate,
            max_depth=xgb.max_depth,
            min_child_weight=xgb.min_child_weight,
            subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
            gamma=xgb.gamma,
            reg_lambda=xgb.reg_lambda,
            reg_alpha=xgb.reg_alpha,
            max_delta_step=xgb.max_delta_step,
            early_stopping_rounds=xgb.early_stopping_rounds,
            eval_metric=["logloss", "error"] if task == "binary" else ["mlogloss", "merror"],
            tree_method="hist",
            n_jobs=xgb.n_jobs,
            random_state=seed,
        )
    raise ValueError(f"unknown baseline model: {name}")


def make_sampler(strategy: ImbalanceStrategy, seed: int) -> Sampler | None:
    """Resampler for the training split, or None when no resampling is used."""
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    if strategy == "oversample":
        return RandomOverSampler(random_state=seed)
    if strategy == "smote":
        return SMOTE(random_state=seed)
    return None


@dataclass
class FittedBaseline:
    """A fitted ``preprocess -> model`` pipeline plus its decision threshold."""

    pipeline: Pipeline
    threshold: float | None  # binary only; None means argmax / 0.5
    train_rows_after_resampling: int
    best_iteration: int | None

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Class probabilities, shape (n_rows, n_classes)."""
        return self.pipeline.predict_proba(df)

    def predict(self, df: pd.DataFrame, scores: np.ndarray | None = None) -> np.ndarray:
        scores = self.predict_scores(df) if scores is None else scores
        if self.threshold is not None:
            return (scores[:, 1] >= self.threshold).astype(np.int64)
        return scores.argmax(axis=1)


def fit_baseline(
    *,
    model_name: BaselineModel,
    task: Task,
    imbalance: ImbalanceStrategy,
    seed: int,
    cfg: BaselineConfig,
    data_cfg: TonIoTConfig,
    train: pd.DataFrame,
    y_train: np.ndarray,
    val: pd.DataFrame,
    y_val: np.ndarray,
) -> FittedBaseline:
    """Fit preprocessing on train, handle imbalance, fit the model, tune the threshold.

    Resampling and sample weights apply to the training split only. Val is used for
    XGBoost early stopping and threshold tuning, never for fitting the preprocessor.
    """
    preprocessor = make_preprocessor(data_cfg).fit(train)
    x_train = preprocessor.transform(train).astype(np.float32)
    x_val = preprocessor.transform(val).astype(np.float32)

    sampler = make_sampler(imbalance, seed)
    if sampler is not None:
        x_train, y_train = sampler.fit_resample(x_train, y_train)
    weights = None
    power = getattr(cfg, model_name).class_weight_power
    # power 0 means no weighting; passing all-ones weights would still change
    # Random Forest's bootstrap sampling, so pass None to make it exact.
    if imbalance == "class_weight" and power > 0:
        weights = compute_sample_weight("balanced", y_train) ** power

    model = make_estimator(model_name, task, cfg, seed)
    if isinstance(model, XGBClassifier):
        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(x_train, y_train, sample_weight=weights)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    threshold = None
    if task == "binary" and cfg.tune_threshold:
        tuner = TunedThresholdClassifierCV(FrozenEstimator(pipeline), scoring="f1")
        threshold = float(tuner.fit(val, y_val).best_threshold_)

    return FittedBaseline(
        pipeline=pipeline,
        threshold=threshold,
        train_rows_after_resampling=len(y_train),
        best_iteration=int(model.best_iteration) if isinstance(model, XGBClassifier) else None,
    )
