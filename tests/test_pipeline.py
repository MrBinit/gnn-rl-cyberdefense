import json

import pandas as pd
import pytest

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.schema import (
    EXCLUDED,
    SPLITS,
    load_clean,
    load_split,
    load_split_sets,
    split_assignment,
)


def test_build_dataset_writes_clean_and_split_sets(tmp_cfg: ExperimentConfig) -> None:
    meta = build_dataset(tmp_cfg)
    out = tmp_cfg.dataset.processed_dir

    assert (out / "clean.parquet").exists()
    assert (out / "splits.parquet").exists()
    assert not any((out / f"{name}.parquet").exists() for name in SPLITS)
    assert json.loads((out / "metadata.json").read_text())["split_sets"] == meta["split_sets"]
    sets = load_split_sets(tmp_cfg.dataset)
    assert len(sets) == len(load_clean(tmp_cfg.dataset))
    assert set(meta["split_sets"]) == set(sets.columns)
    assert {"random", "bot", "bot_unseen_hosts"} <= set(sets.columns)
    assert sum(meta["split_sets"]["random"]["rows"].values()) == meta["clean"]["clean_rows"]

    splits = {name: load_split(tmp_cfg.dataset, name) for name in SPLITS}
    assert sum(len(df) for df in splits.values()) == len(load_clean(tmp_cfg.dataset))
    # No row is in two splits.
    hashes = {n: set(pd.util.hash_pandas_object(df, index=False)) for n, df in splits.items()}
    assert not hashes["train"] & hashes["test"]
    assert not hashes["train"] & hashes["val"]


def test_build_dataset_preserves_dtypes(tmp_cfg: ExperimentConfig) -> None:
    build_dataset(tmp_cfg)
    train = load_split(tmp_cfg.dataset, "train")
    assert isinstance(train["PROTOCOL"].dtype, pd.CategoricalDtype)
    assert train["IN_BYTES"].dtype == "float32"


def test_build_dataset_writes_figures(tmp_cfg: ExperimentConfig) -> None:
    meta = build_dataset(tmp_cfg)
    eda = tmp_cfg.dataset.eda_dir
    assert {p.name for p in eda.glob("*.png")} == {
        "class_balance.png",
        "sparsity.png",
        "classes_per_day.png",
        "src_ip_purity.png",
        "host_degree.png",
        "features_by_class.png",
    }
    assert meta["eda"]["hosts"] > 0
    assert meta["eda"]["days"] == 4
    assert len(meta["days"]) == 4  # one synthetic class per day


def test_figures_can_be_skipped(tmp_cfg: ExperimentConfig) -> None:
    cfg = tmp_cfg.model_copy(
        update={"dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False})}
    )
    assert "eda" not in build_dataset(cfg)
    assert not cfg.dataset.eda_dir.exists()


def test_load_split_follows_the_configured_split_set(tmp_cfg: ExperimentConfig) -> None:
    build_dataset(tmp_cfg)
    cfg = tmp_cfg.dataset.model_copy(update={"split_set": "bot_unseen_hosts"})
    assignment = split_assignment(cfg)
    test = load_split(cfg, "test")
    assert len(test) == (assignment == "test").sum()
    assert set(test["day"]) == {"2018-02-17"}
    kept = sum(len(load_split(cfg, s)) for s in SPLITS)
    assert kept == len(assignment) - (assignment == EXCLUDED).sum()
    with pytest.raises(ValueError, match="unknown split set"):
        split_assignment(cfg, "nope")
