"""Synthetic NetFlow-v3-shaped data so tests run fast and without the real dataset."""

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

from cyberdefense.config import DatasetConfig, ExperimentConfig, PathsConfig, SplitConfig
from cyberdefense.data.netflow import (
    CATEGORICAL_COLS,
    DROPPED,
    NUMERIC_COLS,
    REMOVED_FEATURES,
    TTL_COLS,
)

# Raw "Attack" values; cleaning lower-cases them and maps Benign -> normal.
CLASSES = ["Benign", "ddos", "scanning", "mitm"]
BIG_GROUP_ROWS = 40  # identical-feature rows, larger than the split tests' max_group_size
DAY_MS = 86_400_000
START_MS = 1_518_566_400_000  # 2018-02-14 00:00 UTC
CNC_HOST = "10.0.0.99"


def make_raw(n_per_class: int = 100, seed: int = 0) -> pd.DataFrame:
    """Raw-CSV-shaped NetFlow v3 frame: one day per class, duplicates and one big group."""
    rng = np.random.default_rng(seed)
    n = n_per_class * len(CLASSES)
    attack = np.repeat(CLASSES, n_per_class)
    class_idx = np.repeat(np.arange(len(CLASSES)), n_per_class)

    start = START_MS + class_idx * DAY_MS + rng.integers(0, DAY_MS // 2, n)
    df = pd.DataFrame(
        {
            "FLOW_START_MILLISECONDS": start,
            "FLOW_END_MILLISECONDS": start + rng.integers(0, 5000, n),
            "IPV4_SRC_ADDR": [f"192.168.1.{i}" for i in rng.integers(1, 30, n)],
            "L4_SRC_PORT": rng.integers(1024, 65535, n),
            "IPV4_DST_ADDR": [f"10.0.0.{i}" for i in rng.integers(1, 30, n)],
            "L4_DST_PORT": rng.integers(1, 1024, n),
        }
    )
    for col in NUMERIC_COLS + TTL_COLS + list(DROPPED) + list(REMOVED_FEATURES):
        # Class-dependent scale so models have real signal to learn.
        df[col] = rng.integers(0, 1000, n) * (class_idx + 1)
    for col in CATEGORICAL_COLS:
        df[col] = rng.choice([0, 6, 17], n)
    df["L7_PROTO"] = rng.choice([7.0, 7.178, 91.0], n)
    df["Label"] = (attack != "Benign").astype(int)
    df["Attack"] = attack
    # mitm plays the bot in the bot split sets: 3 infected senders (also active on other
    # days) talking to one C&C host, which attacked earlier (as the real C&C ran GoldenEye).
    df.loc[np.flatnonzero(attack == "ddos")[:10], "IPV4_SRC_ADDR"] = CNC_HOST
    mitm = attack == "mitm"
    df.loc[mitm, "IPV4_SRC_ADDR"] = [f"192.168.1.{i}" for i in rng.integers(1, 4, mitm.sum())]
    df.loc[mitm, "IPV4_DST_ADDR"] = CNC_HOST

    # A big group of identical-feature scanning flows (only the source port differs).
    big = df[df["Attack"] == "scanning"].iloc[[0] * BIG_GROUP_ROWS].copy()
    big["L4_SRC_PORT"] = np.arange(BIG_GROUP_ROWS) + 40000
    # Exact duplicates of the first five rows.
    return pd.concat([df, big, df.iloc[:5]], ignore_index=True)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ExperimentConfig:
    """Config whose every path lives under pytest's tmp_path."""
    raw_csv = tmp_path / "raw" / "NF-test-v3.csv"
    raw_csv.parent.mkdir()
    make_raw().to_csv(raw_csv, index=False)
    return ExperimentConfig(
        paths=PathsConfig(results_dir=tmp_path / "results", tb_dir=tmp_path / "tb"),
        dataset=DatasetConfig(
            raw_csv=raw_csv,
            processed_dir=tmp_path / "processed",
            eda_dir=tmp_path / "eda",
            min_category_freq=2,
            chunk_rows=150,  # exercise chunked reading
            # Keep every synthetic row: all attack-free windows, no per-window attack cap.
            normal_window_ratio=1e6,
            max_attack_flows_per_window=None,
        ),
        # The synthetic big group (40 rows) must be splittable on 400-row data; mitm = bot.
        split=SplitConfig(max_group_size=20, bot_attack_type="mitm", bot_cnc_hosts=[CNC_HOST]),
    )
