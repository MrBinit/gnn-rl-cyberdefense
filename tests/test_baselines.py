import json

import numpy as np
import pandas as pd
import pytest

from cyberdefense.baselines.experiment import encode_targets, iter_specs, run_all
from cyberdefense.baselines.models import fit_baseline, make_sampler
from cyberdefense.config import (
    BaselineConfig,
    ExperimentConfig,
    RandomForestConfig,
    XGBoostConfig,
)
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.schema import SPLITS, load_split
from cyberdefense.reporting import aggregate

SMALL_MODELS = {
    "random_forest": RandomForestConfig(n_estimators=10),
    "xgboost": XGBoostConfig(n_estimators=20, early_stopping_rounds=5),
}


@pytest.fixture
def small_cfg(tmp_cfg: ExperimentConfig) -> ExperimentConfig:
    """Tiny models and prepared synthetic splits, so the grid runs in seconds."""
    cfg = tmp_cfg.model_copy(
        update={
            "baseline": BaselineConfig(**SMALL_MODELS),
            "dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False}),
        }
    )
    build_dataset(cfg)
    return cfg


@pytest.fixture
def splits(small_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    return {name: load_split(small_cfg.dataset, name) for name in SPLITS}


def test_iter_specs_grid() -> None:
    cfg = ExperimentConfig(
        baseline=BaselineConfig(imbalance=["none", "class_weight"], seeds=[1, 2])
    )
    specs = iter_specs(cfg, ["ddos", "mitm"])
    # random: 2 tasks x 2 models x 2 strategies x 2 seeds; unseen: 2 holdouts x 2 x 2 x 2
    assert len(specs) == 16 + 16
    assert not [s for s in specs if s.protocol == "unseen_attack" and s.task == "multiclass"]
    assert len({s.run_name for s in specs}) == len(specs)


@pytest.mark.parametrize("strategy", ["none", "class_weight"])
def test_no_sampler_for_weight_strategies(strategy: str) -> None:
    assert make_sampler(strategy, seed=0) is None


@pytest.mark.parametrize("strategy", ["undersample", "oversample", "smote"])
def test_resampling_balances_train_only(
    small_cfg: ExperimentConfig, splits: dict[str, pd.DataFrame], strategy: str
) -> None:
    classes = sorted(splits["train"]["attack_type"].astype(str).unique())
    y_train = encode_targets(splits["train"], "multiclass", classes)
    y_val = encode_targets(splits["val"], "multiclass", classes)
    fitted = fit_baseline(
        model_name="random_forest",
        task="multiclass",
        imbalance=strategy,
        seed=0,
        cfg=small_cfg.baseline,
        data_cfg=small_cfg.dataset,
        train=splits["train"],
        y_train=y_train,
        val=splits["val"],
        y_val=y_val,
    )
    counts = np.bincount(y_train)
    expected = counts.min() if strategy == "undersample" else counts.max()
    assert fitted.train_rows_after_resampling == expected * len(counts)
    # Prediction works on the untouched val split.
    assert len(fitted.predict(splits["val"])) == len(splits["val"])


def test_threshold_tuned_for_binary_only(
    small_cfg: ExperimentConfig, splits: dict[str, pd.DataFrame]
) -> None:
    common = {
        "model_name": "xgboost",
        "imbalance": "class_weight",
        "seed": 0,
        "cfg": small_cfg.baseline,
        "data_cfg": small_cfg.dataset,
        "train": splits["train"],
        "val": splits["val"],
    }
    classes = sorted(splits["train"]["attack_type"].astype(str).unique())
    binary = fit_baseline(
        task="binary",
        y_train=encode_targets(splits["train"], "binary", classes),
        y_val=encode_targets(splits["val"], "binary", classes),
        **common,
    )
    multi = fit_baseline(
        task="multiclass",
        y_train=encode_targets(splits["train"], "multiclass", classes),
        y_val=encode_targets(splits["val"], "multiclass", classes),
        **common,
    )
    assert binary.threshold is not None
    assert 0 < binary.threshold < 1
    assert multi.threshold is None
    assert binary.best_iteration is not None


def test_run_all_grid(small_cfg: ExperimentConfig, splits: dict[str, pd.DataFrame]) -> None:
    summary = run_all(small_cfg, splits)

    attack_types = 3  # synthetic data: ddos, mitm, scanning
    assert len(summary) == 2 * 2 + attack_types * 2
    for col in ("test_f1", "test_mcc", "test_balanced_accuracy"):
        assert summary[col].between(-1, 1).all()
    binary = summary[summary["task"] == "binary"]
    assert binary["test_pr_auc"].notna().all()
    assert binary["threshold"].notna().all()

    out = small_cfg.run_dir / "baselines"
    assert (out / "summary.csv").exists()
    assert (out / "summary_aggregated.csv").exists()
    assert len(list(out.rglob("*.json"))) == len(summary)
    assert len(list((out / "random").rglob("*__confusion.png"))) == 4

    multi_file = next((out / "random" / "multiclass").glob("*xgboost*.json"))
    multi = json.loads(multi_file.read_text())
    assert set(multi["test_per_class"]) == {"ddos", "mitm", "normal", "scanning"}
    assert np.asarray(multi["test_confusion"]).shape == (4, 4)
    assert set(multi) >= {"train", "val", "test"}

    unseen = summary[summary["protocol"] == "unseen_attack"]
    assert unseen["test_holdout_recall"].between(0, 1).all()
    assert any((small_cfg.paths.tb_dir / "baselines").iterdir())


def test_aggregate_over_seeds() -> None:
    summary = pd.DataFrame(
        {
            "protocol": ["random"] * 2,
            "task": ["binary"] * 2,
            "model": ["xgboost"] * 2,
            "imbalance": ["class_weight"] * 2,
            "holdout": [None, None],
            "seed": [1, 2],
            "test_f1": [0.8, 0.9],
        }
    )
    agg = aggregate(summary)
    assert agg.loc[0, "n_seeds"] == 2
    assert agg.loc[0, "test_f1_mean"] == pytest.approx(0.85)
    assert agg.loc[0, "test_f1_std"] == pytest.approx(np.std([0.8, 0.9], ddof=1))
