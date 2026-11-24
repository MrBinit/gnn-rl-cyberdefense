"""Tune the baselines on the validation split.

Random Forest uses random search; XGBoost uses Optuna TPE with median pruning.
Each search starts with the untuned default as trial 0 for reference.
Usage::

    uv run python scripts/tune_baselines.py                      # split set: random
    uv run python scripts/tune_baselines.py --dataset.split_set bot_unseen_hosts
    uv run python scripts/tune_baselines.py --tuning.models '["xgboost"]' --tuning.xgb_trials 60
    uv run python scripts/tune_baselines.py --tuning.train_fraction 0.5   # faster trials

Then evaluate the tuned models on test with the usual protocols::

    uv run python scripts/train_baselines.py --baseline.use_tuned

Outputs (results/<split_set>/tuning_disjoint/<model>/):
    trials.csv        every trial: params, objective, macro-F1, per-holdout recall, FPR
    best_params.json  best params with their val scores next to the default's
    study.db          (xgboost) Optuna study; rerunning resumes it
TensorBoard: results/<split_set>/tensorboard/tuning/
"""

import logging

import pandas as pd

from cyberdefense.baselines.tuning import tune
from cyberdefense.config import ExperimentConfig
from cyberdefense.data.schema import load_split
from cyberdefense.utils import setup_logging

log = logging.getLogger("tune_baselines")
SHOW = ["objective", "macro_f1", "unseen_recall", "fpr"]


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    train, val = load_split(cfg.dataset, "train"), load_split(cfg.dataset, "val")
    results = tune(cfg, train, val)

    for model_name, best in results.items():
        holdout_cols = [f"unseen_recall_{h}" for h in cfg.tuning.holdouts]
        class_cols = [f"f1_{h}" for h in cfg.tuning.holdouts]
        rows = {"default": best["default_val_scores"], "tuned": best["val_scores"]}
        table = pd.DataFrame({k: v for k, v in rows.items() if v}).T
        table = table[SHOW + class_cols + holdout_cols]
        if "default" in table.index:
            table.loc["change"] = table.loc["tuned"] - table.loc["default"]
        log.info(
            "%s (val, trial %d):\n%s\nbest params: %s",
            model_name,
            best["trial"],
            table.round(4).T.to_string(),
            best["params"],
        )
    log.info("next: uv run python scripts/train_baselines.py --baseline.use_tuned")


if __name__ == "__main__":
    main()
