"""Controlled hyperparameter tuning for the baselines.

- Random Forest: random search (scikit-learn ``ParameterSampler``).
- XGBoost: Bayesian optimisation with Optuna's TPE sampler and median pruning.

Every trial is scored on the **validation split only** (test stays untouched) with::

    objective = w_macro_f1 * val macro-F1 (multiclass, includes mitm)
              + w_unseen   * mean val recall on each tuning holdout, trained without it
              - w_fpr      * mean val false-positive rate of those holdout models

The first trial of each search is the untuned default configuration, so every
result can be read against the current baseline. Edit the search spaces below to
explore other ranges.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import ParameterSampler, train_test_split
from torch.utils.tensorboard import SummaryWriter

from cyberdefense.baselines.experiment import encode_targets
from cyberdefense.baselines.models import fit_baseline
from cyberdefense.config import (
    BaselineModel,
    ExperimentConfig,
    ObjectiveWeights,
    RandomForestConfig,
    XGBoostConfig,
)
from cyberdefense.data.ton_iot import NORMAL_CLASS, TARGET_BINARY, TARGET_MULTI
from cyberdefense.metrics import multiclass_metrics, per_class_report
from cyberdefense.tracking import make_writer

log = logging.getLogger(__name__)

# Random Forest search space (scipy distributions or lists, sampled uniformly).
RF_SPACE = {
    "n_estimators": randint(100, 601),
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 11),
    "max_features": ["sqrt", "log2", 0.2, 0.35, 0.5],
    "max_samples": [None, 0.5, 0.7, 0.9],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight_power": uniform(0.0, 1.0),
}


def suggest_xgb(trial: optuna.Trial) -> dict:
    """XGBoost search space for Optuna (n_estimators stays at the early-stopping cap)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 1.0),
    }


XGB_TUNED = (
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_lambda",
    "reg_alpha",
    "max_delta_step",
    "class_weight_power",
)
# Untuned settings, evaluated as trial 0 of each search for reference.
XGB_DEFAULTS = {k: getattr(XGBoostConfig(), k) for k in XGB_TUNED} | {
    "max_delta_step": int(XGBoostConfig().max_delta_step)
}
RF_DEFAULTS = {k: getattr(RandomForestConfig(), k) for k in RF_SPACE}

ReportFn = Callable[[int, float], None]


@dataclass(frozen=True)
class TuningData:
    train: pd.DataFrame
    val: pd.DataFrame
    classes: list[str]


def prepare_tuning_data(
    cfg: ExperimentConfig, train: pd.DataFrame, val: pd.DataFrame
) -> TuningData:
    classes = sorted(train[TARGET_MULTI].astype(str).unique())
    if missing := set(cfg.tuning.holdouts) - (set(classes) - {NORMAL_CLASS}):
        raise ValueError(f"tuning holdouts are not attack types in the data: {sorted(missing)}")
    if cfg.tuning.train_fraction < 1:
        train, _ = train_test_split(
            train,
            train_size=cfg.tuning.train_fraction,
            stratify=train[TARGET_MULTI],
            random_state=cfg.seed,
        )
    return TuningData(train=train, val=val, classes=classes)


def objective_value(components: dict[str, float], weights: ObjectiveWeights) -> float:
    return (
        weights.macro_f1 * components["macro_f1"]
        + weights.unseen_recall * components["unseen_recall"]
        - weights.fpr * components["fpr"]
    )


def evaluate_params(
    model_name: BaselineModel,
    params: dict,
    cfg: ExperimentConfig,
    data: TuningData,
    report: ReportFn | None = None,
) -> dict[str, float]:
    """Score one hyperparameter setting on val. Returns the objective and its parts."""
    model_cfg = getattr(cfg.baseline, model_name)
    model_cfg = type(model_cfg).model_validate(model_cfg.model_dump() | params)
    bcfg = cfg.baseline.model_copy(update={model_name: model_cfg})
    common = {
        "model_name": model_name,
        "imbalance": "class_weight",  # its strength is tuned via class_weight_power
        "seed": cfg.seed,
        "cfg": bcfg,
        "data_cfg": cfg.ton_iot,
    }
    train, val, classes = data.train, data.val, data.classes
    start = time.perf_counter()

    # 1. Multiclass on the full training split: macro-F1 and the target classes.
    y_val_multi = encode_targets(val, "multiclass", classes)
    fitted = fit_baseline(
        task="multiclass",
        train=train,
        y_train=encode_targets(train, "multiclass", classes),
        val=val,
        y_val=y_val_multi,
        **common,
    )
    pred = fitted.predict(val)
    components = {"macro_f1": multiclass_metrics(y_val_multi, pred)["f1"]}
    per_class = per_class_report(y_val_multi, pred, classes)
    for name in cfg.tuning.holdouts:
        components[f"f1_{name}"] = per_class[name]["f1-score"]

    # 2. One binary model per holdout, trained (and threshold-tuned) without it.
    recalls, fprs = [], []
    is_normal = (val[TARGET_MULTI] == NORMAL_CLASS).to_numpy()
    for step, holdout in enumerate(cfg.tuning.holdouts, start=1):
        keep_train = train[TARGET_MULTI] != holdout
        keep_val = val[TARGET_MULTI] != holdout
        fitted = fit_baseline(
            task="binary",
            train=train[keep_train],
            y_train=train.loc[keep_train, TARGET_BINARY].to_numpy(dtype=np.int64),
            val=val[keep_val],
            y_val=val.loc[keep_val, TARGET_BINARY].to_numpy(dtype=np.int64),
            **common,
        )
        pred = fitted.predict(val)
        recall = float(pred[~keep_val.to_numpy()].mean())
        fpr = float(pred[is_normal].mean())
        components[f"unseen_recall_{holdout}"] = recall
        components[f"fpr_{holdout}"] = fpr
        recalls.append(recall)
        fprs.append(fpr)
        if report is not None:
            partial = {**components, "unseen_recall": np.mean(recalls), "fpr": np.mean(fprs)}
            report(step, objective_value(partial, cfg.tuning.weights))

    components["unseen_recall"] = float(np.mean(recalls))
    components["fpr"] = float(np.mean(fprs))
    components["objective"] = objective_value(components, cfg.tuning.weights)
    components["seconds"] = time.perf_counter() - start
    return components


def _log_trial(writer: SummaryWriter, step: int, result: dict, best: float) -> None:
    for key, value in result.items():
        if isinstance(value, float) and key != "seconds":
            writer.add_scalar(f"tuning/{key}", value, step)
    writer.add_scalar("tuning/best_objective", best, step)


def _describe(result: dict, holdouts: list[str]) -> str:
    unseen = " ".join(f"{h}={result[f'unseen_recall_{h}']:.3f}" for h in holdouts)
    return (
        f"objective={result['objective']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"unseen_recall={result['unseen_recall']:.4f} ({unseen}) "
        f"fpr={result['fpr']:.4f} [{result['seconds']:.0f}s]"
    )


def _clean(params: dict) -> dict:
    """numpy scalars -> plain Python, so params serialise and validate."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()}


def random_search_rf(cfg: ExperimentConfig, data: TuningData, out_dir: Path) -> pd.DataFrame:
    """Random search over ``RF_SPACE``; trial 0 is the untuned default."""
    tcfg = cfg.tuning
    candidates = [RF_DEFAULTS] + [
        _clean(p)
        for p in ParameterSampler(RF_SPACE, n_iter=tcfg.rf_trials - 1, random_state=cfg.seed)
    ]
    writer = make_writer(cfg.paths.tb_dir, "tuning", "random_forest", tcfg.model_dump_json())
    deadline = time.monotonic() + tcfg.timeout_minutes * 60 if tcfg.timeout_minutes else None
    rows, best = [], -np.inf
    for i, params in enumerate(candidates):
        if deadline and time.monotonic() > deadline:
            log.info("timeout reached after %d trials", i)
            break
        result = evaluate_params("random_forest", params, cfg, data)
        best = max(best, result["objective"])
        label = "default" if i == 0 else f"{i}/{len(candidates) - 1}"
        log.info("[rf %s] %s | %s", label, _describe(result, tcfg.holdouts), params)
        rows.append(
            {
                "trial": i,
                "is_default": i == 0,
                **result,
                **{f"param_{k}": v for k, v in params.items()},
            }
        )
        _log_trial(writer, i, result, best)
        pd.DataFrame(rows).to_csv(out_dir / "trials.csv", index=False)  # crash-safe progress
    writer.close()
    return pd.DataFrame(rows)


def optuna_search_xgb(cfg: ExperimentConfig, data: TuningData, out_dir: Path) -> pd.DataFrame:
    """TPE search over ``suggest_xgb``; trial 0 is the untuned default. Resumable."""
    tcfg = cfg.tuning
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="xgboost",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True),
        pruner=(
            optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=tcfg.prune_after_holdouts - 1
            )
            if tcfg.prune
            else optuna.pruners.NopPruner()
        ),
    )
    if not study.trials:
        study.enqueue_trial(XGB_DEFAULTS, user_attrs={"is_default": True})
    else:
        log.info(
            "resuming study with %d existing trials (delete %s to restart)",
            len(study.trials),
            out_dir / "study.db",
        )
    writer = make_writer(cfg.paths.tb_dir, "tuning", "xgboost", tcfg.model_dump_json())

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb(trial)

        def report(step: int, value: float) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned

        result = evaluate_params("xgboost", params, cfg, data, report)
        for key, value in result.items():
            trial.set_user_attr(key, value)
        return result["objective"]

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        label = "default" if trial.user_attrs.get("is_default") else str(trial.number)
        if trial.state == optuna.trial.TrialState.PRUNED:
            log.info("[xgb %s] pruned at step %s | %s", label, trial.last_step, trial.params)
            return
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log.info(
                "[xgb %s] %s | %s", label, _describe(trial.user_attrs, tcfg.holdouts), trial.params
            )
            _log_trial(writer, trial.number, trial.user_attrs, study.best_value)

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = tcfg.xgb_trials - len(study.get_trials(deepcopy=False, states=finished))
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[on_trial_end],
            timeout=tcfg.timeout_minutes * 60 if tcfg.timeout_minutes else None,
        )
    else:
        log.info(
            "xgboost budget of %d trials already used; raise --tuning.xgb_trials to continue",
            tcfg.xgb_trials,
        )
    writer.close()

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        attrs = {k: v for k, v in t.user_attrs.items() if k != "is_default"}
        rows.append(
            {
                "trial": t.number,
                "is_default": bool(t.user_attrs.get("is_default")),
                **attrs,
                **{f"param_{k}": v for k, v in t.params.items()},
            }
        )
    trials = pd.DataFrame(rows)
    trials.to_csv(out_dir / "trials.csv", index=False)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    log.info("xgboost: %d complete, %d pruned", len(trials), n_pruned)
    return trials


def save_best(
    trials: pd.DataFrame, model_name: BaselineModel, cfg: ExperimentConfig, out_dir: Path
) -> dict:
    """Write best_params.json (best objective) and return it."""
    best = trials.loc[trials["objective"].idxmax()]
    default = trials[trials["is_default"]]
    params = {k.removeprefix("param_"): v for k, v in best.items() if k.startswith("param_")}
    params = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in params.items()}
    metric_cols = [
        c for c in trials.columns if not c.startswith("param_") and c not in ("trial", "is_default")
    ]
    payload = {
        "model": model_name,
        "trial": int(best["trial"]),
        "params": _clean(params),
        "val_scores": {c: float(best[c]) for c in metric_cols},
        "default_val_scores": (
            {c: float(default.iloc[0][c]) for c in metric_cols} if not default.empty else None
        ),
        "tuning_config": json.loads(cfg.tuning.model_dump_json()),
    }
    (out_dir / "best_params.json").write_text(json.dumps(payload, indent=2))
    return payload


# Settings that define the objective; a folder must never mix trials scored differently.
OBJECTIVE_FIELDS = ("holdouts", "weights", "train_fraction")


def check_objective_unchanged(cfg: ExperimentConfig, model_dir: Path) -> None:
    """Refuse to reuse a tuning folder whose earlier results used another objective."""
    path = model_dir / "best_params.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())["tuning_config"]
    current = json.loads(cfg.tuning.model_dump_json())
    changed = [f for f in OBJECTIVE_FIELDS if previous.get(f) != current[f]]
    if changed:
        raise ValueError(
            f"{model_dir} holds results tuned with different {changed}: "
            f"{ {f: previous.get(f) for f in changed} }. Use another --tuning.out_dir "
            "or delete that folder."
        )


def tune(cfg: ExperimentConfig, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, dict]:
    """Tune every model in ``cfg.tuning.models``; return {model: best_params payload}."""
    data = prepare_tuning_data(cfg, train, val)
    out = {}
    for model_name in cfg.tuning.models:
        out_dir = cfg.tuning.out_dir / model_name
        check_objective_unchanged(cfg, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "tuning %s on %s train rows, holdouts %s",
            model_name,
            f"{len(data.train):,}",
            cfg.tuning.holdouts,
        )
        search = random_search_rf if model_name == "random_forest" else optuna_search_xgb
        trials = search(cfg, data, out_dir)
        out[model_name] = save_best(trials, model_name, cfg, out_dir)
    return out


def apply_tuned(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a config with the tuned best params and the holdouts actually tuned on.

    The holdouts come from the saved results, not the CLI, so "tuned for" flags in
    the evaluation always match what the search optimised.
    """
    updates, tuned_holdouts = {}, set()
    for model_name in cfg.baseline.models:
        path = cfg.tuning.out_dir / model_name / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/tune_baselines.py first")
        saved = json.loads(path.read_text())
        current = getattr(cfg.baseline, model_name)
        updates[model_name] = type(current).model_validate(current.model_dump() | saved["params"])
        tuned_holdouts |= set(saved["tuning_config"]["holdouts"])
    return cfg.model_copy(
        update={
            "baseline": cfg.baseline.model_copy(update=updates),
            "tuning": cfg.tuning.model_copy(update={"holdouts": sorted(tuned_holdouts)}),
        }
    )
