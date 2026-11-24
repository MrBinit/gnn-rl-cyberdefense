"""Column names and split access shared by every stage, independent of the source dataset.

``data/processed/<dataset>/clean.parquet`` holds every prepared row; ``splits.parquet`` is
row-aligned with it and has one column per split set (``random``, ``lodo_<day>``, ``bot``,
``bot_unseen_hosts``) holding train / val / test / excluded. The prepared table always
carries these columns, whatever the raw format: host/port ids (graph only, never features),
``timestamp``, ``day`` and ``window`` (splits and window graphs only, never features), the feature
columns of the dataset module, the targets, ``noisy_label`` and ``feature_group``.
"""

import pandas as pd
import pyarrow.parquet as pq

from cyberdefense.config import DatasetConfig

TARGET_BINARY = "label"  # 0 normal, 1 attack
TARGET_MULTI = "attack_type"  # normal / ddos_attack-hoic / bot / ...
NORMAL_CLASS = "normal"
FEATURE_GROUP = "feature_group"  # hash of the feature vector, used to group splits
NOISY_LABEL = "noisy_label"  # 1 for attack types with documented label problems
TIMESTAMP = "timestamp"  # flow start, seconds since epoch
DAY = "day"  # capture day, YYYY-MM-DD (day-based splits)
WINDOW = "window"  # time-window id = flow start // window_seconds (window graphs)
SPLIT_COL = "split"
SPLITS = ("train", "val", "test")
ID_COLS = ("src_ip", "src_port", "dst_ip", "dst_port")


SPLIT_SETS_FILE = "splits.parquet"
EXCLUDED = "excluded"  # rows a split set keeps out of every split (e.g. the bot C&C history)


def load_clean(cfg: DatasetConfig) -> pd.DataFrame:
    """All prepared rows (built by scripts/prepare_data.py), without split assignment."""
    return pd.read_parquet(cfg.processed_dir / "clean.parquet")


def load_split_sets(cfg: DatasetConfig) -> pd.DataFrame:
    """Every split set, one column each, row-aligned with ``load_clean``."""
    return pd.read_parquet(cfg.processed_dir / SPLIT_SETS_FILE)


def split_assignment(cfg: DatasetConfig, split_set: str | None = None) -> pd.Series:
    """train / val / test / excluded per prepared row for one split set (default: configured)."""
    name = split_set or cfg.split_set
    path = cfg.processed_dir / SPLIT_SETS_FILE
    available = pq.read_schema(path).names
    if name not in available:
        raise ValueError(f"unknown split set {name!r}; available: {available}")
    return pd.read_parquet(path, columns=[name])[name].astype(str)


def load_split(cfg: DatasetConfig, split: str) -> pd.DataFrame:
    """train / val / test of the configured split set (``dataset.split_set``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    mask = (split_assignment(cfg) == split).to_numpy()
    return load_clean(cfg)[mask].reset_index(drop=True)
