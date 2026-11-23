"""Synthetic ToN-IoT-shaped data so tests run fast and without the real dataset."""

from pathlib import Path

# The suite runs baselines (xgboost) and graph code (torch) in one process: load xgboost
# first and keep torch on one thread, or the two OpenMP runtimes segfault (see
# cyberdefense/__init__). Real runs keep them in separate processes.
import xgboost  # noqa: F401  # isort: skip
import torch  # isort: skip

torch.set_num_threads(1)

import numpy as np
import pandas as pd
import pytest

from cyberdefense.config import ExperimentConfig, PathsConfig, SplitConfig, TonIoTConfig
from cyberdefense.data.ton_iot import CATEGORICAL_COLS, MISSING, NUMERIC_COLS

CLASSES = ["normal", "ddos", "scanning", "mitm"]
TEXT_COLS = TonIoTConfig().text_cols
CATEGORY_VALUES = ["a", "b", MISSING]
BIG_GROUP_ROWS = 40  # identical-feature rows, larger than the split tests' max_group_size


def make_raw(n_per_class: int = 100, seed: int = 0) -> pd.DataFrame:
    """Raw-CSV-shaped frame: '-' placeholders, text columns, duplicates, one big group."""
    rng = np.random.default_rng(seed)
    n = n_per_class * len(CLASSES)
    attack_type = np.repeat(CLASSES, n_per_class)
    class_idx = np.repeat(np.arange(len(CLASSES)), n_per_class)

    df = pd.DataFrame(
        {
            "src_ip": [f"192.168.1.{i}" for i in rng.integers(1, 30, n)],
            "src_port": rng.integers(1024, 65535, n),
            "dst_ip": [f"10.0.0.{i}" for i in rng.integers(1, 30, n)],
            "dst_port": rng.integers(1, 1024, n),
        }
    )
    for col in NUMERIC_COLS:
        # Class-dependent scale so models have real signal to learn.
        df[col] = rng.integers(0, 1000, n) * (class_idx + 1)
    df["http_trans_depth"] = rng.choice(["1", "2", MISSING], n)
    for col in CATEGORICAL_COLS:
        df[col] = rng.choice(CATEGORY_VALUES, n)
    for col in TEXT_COLS:
        df[col] = [f"text-{i}" for i in rng.integers(0, 10**6, n)]
    df["label"] = (attack_type != "normal").astype(int)
    df["type"] = attack_type

    # A big group of identical-feature scanning flows (only ids differ).
    big = df[df["type"] == "scanning"].iloc[[0] * BIG_GROUP_ROWS].copy()
    big["src_port"] = np.arange(BIG_GROUP_ROWS) + 40000
    # Exact duplicates of the first five rows.
    return pd.concat([df, big, df.iloc[:5]], ignore_index=True)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ExperimentConfig:
    """Config whose every path lives under pytest's tmp_path."""
    raw_csv = tmp_path / "raw" / "train_test_network.csv"
    raw_csv.parent.mkdir()
    make_raw().to_csv(raw_csv, index=False)
    return ExperimentConfig(
        paths=PathsConfig(results_dir=tmp_path / "results", tb_dir=tmp_path / "tb"),
        ton_iot=TonIoTConfig(
            raw_csv=raw_csv,
            processed_dir=tmp_path / "processed",
            eda_dir=tmp_path / "eda",
            min_category_freq=2,
        ),
        # The synthetic big group (40 rows) must be splittable on 400-row data.
        split=SplitConfig(max_group_size=20),
    )
