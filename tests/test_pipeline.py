import json

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.ton_iot import SPLITS, load_clean, load_split


def test_build_dataset_writes_parquet_splits(tmp_cfg: ExperimentConfig) -> None:
    meta = build_dataset(tmp_cfg)
    out = tmp_cfg.ton_iot.processed_dir

    for name in ("clean", *SPLITS):
        assert (out / f"{name}.parquet").exists()
    assert json.loads((out / "metadata.json").read_text())["rows"] == meta["rows"]

    splits = {name: load_split(tmp_cfg.ton_iot, name) for name in SPLITS}
    assert sum(len(df) for df in splits.values()) == len(load_clean(tmp_cfg.ton_iot))
    assert sum(meta["rows"].values()) == meta["clean"]["clean_rows"]
    # No row is in two splits.
    hashes = {n: set(pd.util.hash_pandas_object(df, index=False)) for n, df in splits.items()}
    assert not hashes["train"] & hashes["test"]
    assert not hashes["train"] & hashes["val"]


def test_build_dataset_preserves_dtypes(tmp_cfg: ExperimentConfig) -> None:
    build_dataset(tmp_cfg)
    train = load_split(tmp_cfg.ton_iot, "train")
    assert isinstance(train["proto"].dtype, pd.CategoricalDtype)
    assert train["duration"].dtype == "float32"


def test_build_dataset_writes_figures(tmp_cfg: ExperimentConfig) -> None:
    meta = build_dataset(tmp_cfg)
    eda = tmp_cfg.ton_iot.eda_dir
    assert {p.name for p in eda.glob("*.png")} == {
        "class_balance.png",
        "sparsity.png",
        "src_ip_purity.png",
        "host_degree.png",
        "features_by_class.png",
    }
    assert meta["eda"]["hosts"] > 0


def test_figures_can_be_skipped(tmp_cfg: ExperimentConfig) -> None:
    cfg = tmp_cfg.model_copy(
        update={"ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False})}
    )
    assert "eda" not in build_dataset(cfg)
    assert not cfg.ton_iot.eda_dir.exists()
