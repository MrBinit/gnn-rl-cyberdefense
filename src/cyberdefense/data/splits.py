"""Split sets: stratified random, leave-one-attack-day-out and bot generalisation.

``make_split_sets`` returns one column per split set (train / val / test / excluded per
row); ``pipeline`` stores them row-aligned with ``clean.parquet``. The random set:

- Stratified on ``attack_type``: every class (including the rare mitm) keeps its
  share in every split, which also stratifies the binary label.
- Grouped on ``feature_group``: small groups of flows with an identical feature
  vector land in the same split. Groups larger than ``max_group_size`` are common
  traffic patterns (one 1-SYN scanning vector covers 77% of scanning) and are split
  row-wise; forcing them into a single split makes stratification impossible.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cyberdefense.config import SplitConfig
from cyberdefense.data.schema import (
    DAY,
    EXCLUDED,
    FEATURE_GROUP,
    NOISY_LABEL,
    SPLITS,
    TARGET_BINARY,
    TARGET_MULTI,
)

SPLIT_VALUES = (*SPLITS, EXCLUDED)


@dataclass(frozen=True)
class SplitReport:
    group_overlap: dict[str, int]
    max_class_share_deviation: float
    test_rows_with_vector_seen_in_train: float
    class_share: pd.DataFrame


def split_groups(df: pd.DataFrame, cfg: SplitConfig) -> np.ndarray:
    """Group id per row; rows of oversized groups each get their own id."""
    groups, uniques = pd.factorize(df[FEATURE_GROUP])
    if cfg.max_group_size is not None:
        size = np.bincount(groups)[groups]
        big = size > cfg.max_group_size
        groups[big] = len(uniques) + np.flatnonzero(big)  # ids past the last real group
    return groups


def stratified_group_split(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Return a Series aligned with ``df`` holding 'train' / 'val' / 'test' per row.

    Vectorised stratified group k-fold (``StratifiedGroupKFold`` loops over millions of
    groups far too slowly): each group gets its majority class; within each class the
    groups are shuffled (seeded) and cut into ``n_folds`` consecutive chunks of equal row
    count, so every class keeps its share in every fold and no group is split.
    """
    if cfg.val_folds + cfg.test_folds >= cfg.n_folds:
        raise ValueError("val_folds + test_folds must leave at least one fold for train")

    groups, _ = pd.factorize(split_groups(df, cfg))  # compact ids 0..n_groups-1
    classes = pd.Categorical(df[TARGET_MULTI]).codes
    n_groups = int(groups.max()) + 1
    # Majority class per group (groups whose vector occurs with several labels exist).
    pairs = pd.DataFrame({"g": groups, "c": classes}).value_counts().reset_index(name="n")
    pairs = pairs.sort_values(["g", "n"], ascending=[True, False]).drop_duplicates("g")
    group_class = np.full(n_groups, -1, dtype=np.int64)
    group_class[pairs["g"].to_numpy()] = pairs["c"].to_numpy()
    group_size = np.bincount(groups, minlength=n_groups)

    rng = np.random.default_rng(cfg.seed)
    order = np.lexsort((rng.random(n_groups), group_class))  # by class, random within
    size_sorted = group_size[order]
    cls_sorted = group_class[order]
    class_total = np.bincount(cls_sorted, weights=size_sorted)
    class_start = np.r_[0, np.cumsum(class_total)[:-1]]
    rows_before = np.cumsum(size_sorted) - size_sorted - class_start[cls_sorted]
    position = rows_before / np.maximum(class_total[cls_sorted], 1)  # in [0, 1)
    group_fold = np.empty(n_groups, dtype=np.int16)
    group_fold[order] = np.minimum((position * cfg.n_folds).astype(np.int16), cfg.n_folds - 1)
    # Shuffle which fold numbers become test/val, so the chunk order carries no meaning.
    fold_id = rng.permutation(cfg.n_folds)[group_fold[groups]]

    split = np.full(len(df), "train", dtype=object)
    split[fold_id < cfg.test_folds] = "test"
    split[(fold_id >= cfg.test_folds) & (fold_id < cfg.test_folds + cfg.val_folds)] = "val"
    return pd.Series(split, index=df.index, name="split")


def clean_attack_days(df: pd.DataFrame) -> list[str]:
    """Days with reliably labelled (non-noisy) attacks, in order."""
    clean_attack = (df[TARGET_BINARY] == 1) & (df[NOISY_LABEL] == 0)
    return sorted(df.loc[clean_attack, DAY].unique().tolist())


def day_split(df: pd.DataFrame, val_days: list[str], test_days: list[str]) -> np.ndarray:
    """Whole days: ``test_days`` -> test, ``val_days`` -> val, every other day -> train."""
    if not test_days:
        raise ValueError("a day split needs at least one test day")
    if overlap := set(val_days) & set(test_days):
        raise ValueError(f"days in both val and test: {sorted(overlap)}")
    if missing := (set(val_days) | set(test_days)) - set(df[DAY].unique()):
        raise ValueError(f"split days not in the data: {sorted(missing)}")
    split = np.full(len(df), "train", dtype=object)
    split[df[DAY].isin(val_days).to_numpy()] = "val"
    split[df[DAY].isin(test_days).to_numpy()] = "test"
    return split


def lodo_splits(df: pd.DataFrame, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """``lodo_<day>``: test = one clean-attack day, val = the next one (cyclic)."""
    attack_days = clean_attack_days(df)
    days = cfg.lodo_days or attack_days
    if bad := set(days) - set(attack_days):
        raise ValueError(f"lodo days without clean attacks: {sorted(bad)}")
    if len(attack_days) < 3:
        raise ValueError("leave-one-day-out needs at least 3 clean-attack days")
    out = {}
    for day in days:
        val = attack_days[(attack_days.index(day) + 1) % len(attack_days)]
        out[f"lodo_{day}"] = day_split(df, [val], [day])
    return out


def _touches(df: pd.DataFrame, hosts: list[str]) -> np.ndarray:
    return (df["src_ip"].isin(hosts) | df["dst_ip"].isin(hosts)).to_numpy()


def bot_hosts(df: pd.DataFrame, cfg: SplitConfig) -> list[str]:
    """Infected hosts: senders of bot flows, minus the C&C."""
    bot = df[TARGET_MULTI] == cfg.bot_attack_type
    return sorted(set(df.loc[bot, "src_ip"].astype(str)) - set(cfg.bot_cnc_hosts))


def bot_splits(df: pd.DataFrame, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """``bot`` and ``bot_unseen_hosts`` (see ``SplitConfig``)."""
    bot = df[TARGET_MULTI] == cfg.bot_attack_type
    if not bot.any():
        raise ValueError(f"no {cfg.bot_attack_type!r} flows; set split.bot_attack_type")
    test_days = sorted(df.loc[bot, DAY].unique().tolist())
    val_day = cfg.bot_val_day
    if val_day is None:
        earlier = [d for d in clean_attack_days(df) if d < test_days[0]]
        if not earlier:
            raise ValueError("no clean-attack day before the bot day; set split.bot_val_day")
        val_day = earlier[-1]
    base = day_split(df, [val_day], test_days)

    held = base != "test"
    with_cnc = base.copy()
    with_cnc[held & _touches(df, cfg.bot_cnc_hosts)] = EXCLUDED
    unseen = with_cnc.copy()
    unseen[held & _touches(df, bot_hosts(df, cfg))] = EXCLUDED
    return {"bot": with_cnc, "bot_unseen_hosts": unseen}


def make_split_sets(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Every split set, one categorical column each, aligned with ``df``."""
    sets = {"random": stratified_group_split(df, cfg).to_numpy()}
    sets |= lodo_splits(df, cfg)
    sets |= bot_splits(df, cfg)
    dtype = pd.CategoricalDtype(SPLIT_VALUES)
    return pd.DataFrame({k: pd.Categorical(v, dtype=dtype) for k, v in sets.items()})


def check_split(
    df: pd.DataFrame, split: pd.Series, cfg: SplitConfig, enforce_groups: bool = True
) -> SplitReport:
    """Verify group integrity and stratification.

    With ``enforce_groups`` (the random set), raises if a kept feature group crosses
    splits. For day-based sets identical vectors legitimately recur on other days and
    classes differ by design, so both are only reported.
    """
    groups = pd.Series(split_groups(df, cfg), index=df.index)
    members = {s: set(groups[split == s]) for s in SPLITS}
    overlap = {
        "train&val": len(members["train"] & members["val"]),
        "train&test": len(members["train"] & members["test"]),
        "val&test": len(members["val"] & members["test"]),
    }
    if any(overlap.values()) and enforce_groups:
        raise AssertionError(f"feature groups leak across splits: {overlap}")

    share = pd.crosstab(df[TARGET_MULTI], split, normalize="columns").reindex(columns=SPLITS)
    overall = df[TARGET_MULTI].value_counts(normalize=True)
    train_vectors = set(df.loc[split == "train", FEATURE_GROUP])
    return SplitReport(
        group_overlap=overlap,
        max_class_share_deviation=float(share.sub(overall, axis=0).abs().max().max()),
        test_rows_with_vector_seen_in_train=float(
            df.loc[split == "test", FEATURE_GROUP].isin(train_vectors).mean()
        ),
        class_share=share,
    )
