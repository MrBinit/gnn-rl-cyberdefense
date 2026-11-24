"""Dataset validation: provenance of the NetFlow copy, duplicate leakage, split integrity.

Hard checks (any failure makes the report fail):
- the raw CSV has the expected schema (enforced by ``clean``);
- preparation: the prepared table is reproducible from the raw file; no identifier, time,
  TTL or removed column is a feature; features are finite; ``noisy_label`` matches the
  noisy types; kept windows hold their complete (deduplicated) normal traffic, attacks
  are capped per (window, type) and window features match the complete traffic;
- no exact duplicate rows remain after cleaning;
- random set: no row in two splits, no kept-intact feature group crosses splits, every
  class has the same share in train/val/test (within 1 point);
- every split set: train/val/test non-empty, clean attacks in test; bot sets: the C&C
  (and in ``bot_unseen_hosts`` the bot hosts) never appear in train/val.

Informational checks:
- class counts against the published NF-CICIDS2018-v3 file (``netflow.PUBLISHED_COUNTS``);
- a row-level comparison with the official CSV, when one is supplied;
- how many test attack flows come from attacker hosts already seen in training
  (the reason the unseen-host protocol exists).
"""

import json
import logging

import numpy as np
import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.netflow import (
    PUBLISHED_COUNTS,
    RAW_ATTACK,
    RAW_IDS,
    RAW_LABEL,
    RAW_TIME,
    REMOVED_FEATURES,
    TTL_COLS,
    clean,
    feature_cols,
    load_raw,
    normalise_attack,
    numeric_cols,
)
from cyberdefense.data.schema import (
    DAY,
    EXCLUDED,
    ID_COLS,
    NOISY_LABEL,
    NORMAL_CLASS,
    SPLITS,
    TARGET_BINARY,
    TARGET_MULTI,
    TIMESTAMP,
    WINDOW,
    load_clean,
    load_split_sets,
)
from cyberdefense.data.splits import bot_hosts, check_split

log = logging.getLogger(__name__)

# Class counts of the published file this copy should match (before dedup and sampling).
REPORTED_OFFICIAL_COUNTS = PUBLISHED_COUNTS
MAX_CLASS_SHARE_DEVIATION = 0.01


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    """Content hash per row over every column (as text, so dtypes don't matter)."""
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def compare_with_reported(raw: pd.DataFrame) -> dict:
    """Per-class counts of this copy against the reported official counts."""
    counts = normalise_attack(raw[RAW_ATTACK]).value_counts()
    rows = {
        cls: {"this_copy": int(counts.get(cls, 0)), "reported_official": n}
        for cls, n in REPORTED_OFFICIAL_COUNTS.items()
    }
    differing = {c: r for c, r in rows.items() if r["this_copy"] != r["reported_official"]}
    attack_match = all(c == "normal" for c in differing)
    return {
        "per_class": rows,
        "this_copy_rows": len(raw),
        "reported_official_rows": sum(REPORTED_OFFICIAL_COUNTS.values()),
        "differing_classes": sorted(differing),
        "attack_counts_match": attack_match,
        "verdict": (
            "matches the published NF-CICIDS2018-v3 counts"
            if not differing
            else "attack counts match; normal traffic differs (likely a normal-class subsample)"
            if attack_match
            else "differs from the published counts beyond the normal class (other file?)"
        ),
    }


def compare_with_official(raw: pd.DataFrame, official: pd.DataFrame) -> dict:
    """Row-level comparison with the official CSV."""
    same_columns = list(raw.columns) == list(official.columns)
    report: dict = {
        "this_copy_rows": len(raw),
        "official_rows": len(official),
        "same_columns": same_columns,
    }
    if not same_columns:
        report["column_difference"] = sorted(set(raw.columns) ^ set(official.columns))
        return report

    official_hashes = set(_row_hashes(official))
    contained = _row_hashes(raw).isin(official_hashes).to_numpy()
    is_normal = (raw[RAW_LABEL] == 0).to_numpy()
    report |= {
        "share_of_rows_found_in_official": float(contained.mean()),
        "share_of_attack_rows_found": float(contained[~is_normal].mean()),
        "share_of_normal_rows_found": float(contained[is_normal].mean()),
        "official_class_counts": normalise_attack(official[RAW_ATTACK]).value_counts().to_dict(),
        "this_copy_class_counts": normalise_attack(raw[RAW_ATTACK]).value_counts().to_dict(),
    }
    report["this_copy_is_subset"] = report["share_of_rows_found_in_official"] == 1.0
    return report


def check_preparation(raw: pd.DataFrame, fresh: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """Verify cleaning, feature selection and window sampling of the prepared table.

    ``fresh`` is ``clean(raw)`` recomputed now; the stored table must equal it. Window
    completeness is checked by re-cleaning the raw rows of the kept windows without any
    sampling: that is the complete deduplicated traffic of those windows.
    """
    dcfg = cfg.dataset
    df = load_clean(dcfg)
    feats = feature_cols(dcfg)
    never = {*RAW_IDS, *RAW_IDS.values(), *RAW_TIME, "DNS_QUERY_ID", TIMESTAMP, DAY, WINDOW}
    never |= set(REMOVED_FEATURES) | (set() if dcfg.use_ttl_features else set(TTL_COLS))
    nums = df[numeric_cols(dcfg)].to_numpy()

    full_cfg = dcfg.model_copy(
        update={
            "sampling": "windows",
            "normal_window_ratio": 1e9,
            "max_attack_flows_per_window": None,
        }
    )
    start = raw[RAW_TIME[0]].to_numpy(np.float64)
    raw_window = (start // (dcfg.window_seconds * 1000)).astype(np.int64)
    kept_windows = np.unique(df[WINDOW].to_numpy())
    full, _ = clean(raw[np.isin(raw_window, kept_windows)], full_cfg)

    def per_window(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby([WINDOW, TARGET_MULTI], observed=True).size()

    have, complete = per_window(df), per_window(full)
    complete = complete.reindex(complete.index.union(have.index), fill_value=0)
    have = have.reindex(complete.index, fill_value=0)
    normal = complete.index.get_level_values(TARGET_MULTI) == NORMAL_CLASS
    cap = dcfg.max_attack_flows_per_window
    expected = complete.where(normal, complete.clip(upper=cap) if cap else complete)
    full_hashes = set(pd.util.hash_pandas_object(full[list(df.columns)], index=False))
    df_hashes = pd.util.hash_pandas_object(df, index=False)

    clean_attack = (df[TARGET_BINARY] == 1) & (df[NOISY_LABEL] == 0)
    return {
        "reproducible_from_raw": bool(
            len(fresh) == len(df)
            and (
                pd.util.hash_pandas_object(fresh, index=False).to_numpy() == df_hashes.to_numpy()
            ).all()
        ),
        "forbidden_feature_columns": sorted(set(feats) & never),
        "removed_columns_present": sorted(set(REMOVED_FEATURES) & set(df.columns)),
        "identifier_columns_present": sorted(set(ID_COLS) - set(df.columns)) == [],
        "n_features": len(feats),
        "non_finite_feature_values": int((~np.isfinite(nums)).sum()),
        "noisy_flag_mismatches": int(
            (df[NOISY_LABEL].astype(bool) != df[TARGET_MULTI].isin(dcfg.noisy_attack_types)).sum()
        ),
        "window_id_mismatches": int(
            (df[WINDOW] != (np.round(df[TIMESTAMP] * 1000) // (dcfg.window_seconds * 1000))).sum()
        ),
        "windows_kept": len(kept_windows),
        "windows_with_clean_attacks": int(df.loc[clean_attack, WINDOW].nunique()),
        "windows_with_incomplete_normal_traffic": int(
            (have[normal] != complete[normal]).groupby(level=WINDOW).any().sum()
        ),
        "window_type_cells_off_attack_cap": int((have[~normal] != expected[~normal]).sum()),
        "max_attack_flows_per_window_type": int(have[~normal].max()) if (~normal).any() else 0,
        "rows_not_matching_complete_traffic": int((~df_hashes.isin(full_hashes)).sum()),
    }


def check_split_set(df: pd.DataFrame, split: pd.Series, name: str, cfg: ExperimentConfig) -> dict:
    """Integrity of one split set (``split`` holds train / val / test / excluded)."""
    hashes = pd.util.hash_pandas_object(df, index=False)
    members = {s: set(hashes[split == s]) for s in SPLITS}
    row_overlap = {
        "train&val": len(members["train"] & members["val"]),
        "train&test": len(members["train"] & members["test"]),
        "val&test": len(members["val"] & members["test"]),
    }
    report = check_split(df, split, cfg.split, enforce_groups=name == "random")
    attack = df[TARGET_BINARY] == 1
    clean_attack = attack & (df[NOISY_LABEL] == 0)
    train_attackers = set(df.loc[attack & (split == "train"), "src_ip"])
    test_attacks = df[attack & (split == "test")]

    masked: list[str] = []
    if name.startswith("bot"):
        masked = list(cfg.split.bot_cnc_hosts)
        if name == "bot_unseen_hosts":
            masked += bot_hosts(df, cfg.split)
    fit = split.isin(["train", "val"])
    masked_in_fit = (df["src_ip"].isin(masked) | df["dst_ip"].isin(masked)) & fit
    test_types = df.loc[clean_attack & (split == "test"), TARGET_MULTI]
    return {
        "rows": {s: int((split == s).sum()) for s in SPLITS},
        "excluded_rows": int((split == EXCLUDED).sum()),
        "clean_attack_rows": {s: int((clean_attack & (split == s)).sum()) for s in SPLITS},
        "test_attack_types": sorted(test_types.astype(str).unique().tolist()),
        "test_attack_types_also_in_train": sorted(
            set(test_types.astype(str))
            & set(df.loc[attack & (split == "train"), TARGET_MULTI].astype(str))
        ),
        "row_overlap_between_splits": row_overlap,
        "feature_group_overlap_between_splits": report.group_overlap,
        "max_class_share_deviation": report.max_class_share_deviation,
        "test_rows_with_feature_vector_in_train": report.test_rows_with_vector_seen_in_train,
        "test_attack_flows_from_hosts_seen_attacking_in_train": float(
            test_attacks["src_ip"].isin(train_attackers).mean()
        )
        if len(test_attacks)
        else 0.0,
        "masked_hosts": len(masked),
        "masked_host_rows_in_train_val": int(masked_in_fit.sum()),
    }


def check_prepared_splits(cfg: ExperimentConfig) -> dict:
    """Duplicates on the prepared table, then every split set."""
    df = load_clean(cfg.dataset)
    sets = load_split_sets(cfg.dataset)
    if len(sets) != len(df):
        raise ValueError("splits.parquet is not aligned with clean.parquet; rebuild the data")
    duplicates = int(pd.util.hash_pandas_object(df, index=False).duplicated().sum())
    return {
        "rows": len(df),
        "exact_duplicate_rows_after_cleaning": duplicates,
        "noisy_label_rows": int(df[NOISY_LABEL].sum()),
        "split_sets": {
            name: check_split_set(
                df, pd.Series(sets[name].astype(str).to_numpy(), index=df.index), name, cfg
            )
            for name in sets.columns
        },
    }


def hard_failures(prepared: dict, preparation: dict | None = None) -> list[str]:
    """Checks that must pass. Class shares are enforced for the random set only."""
    failures = []
    if prepared["exact_duplicate_rows_after_cleaning"]:
        failures.append(
            f"{prepared['exact_duplicate_rows_after_cleaning']} exact duplicates remain"
        )
    for name, s in prepared["split_sets"].items():
        if any(s["row_overlap_between_splits"].values()):
            failures.append(f"{name}: rows shared between splits {s['row_overlap_between_splits']}")
        if empty := [k for k, n in s["rows"].items() if n == 0]:
            failures.append(f"{name}: empty splits {empty}")
        if not s["clean_attack_rows"]["test"]:
            failures.append(f"{name}: no clean attacks in test")
        if s["masked_host_rows_in_train_val"]:
            failures.append(
                f"{name}: {s['masked_host_rows_in_train_val']} masked-host rows in train/val"
            )
        if name == "random" and s["max_class_share_deviation"] > MAX_CLASS_SHARE_DEVIATION:
            failures.append(f"random: class shares differ by {s['max_class_share_deviation']:.4f}")
    if preparation is not None:
        p = preparation
        must_be_zero = [
            "non_finite_feature_values",
            "noisy_flag_mismatches",
            "window_id_mismatches",
            "windows_with_incomplete_normal_traffic",
            "window_type_cells_off_attack_cap",
            "rows_not_matching_complete_traffic",
        ]
        failures += [f"preparation: {k} = {p[k]}" for k in must_be_zero if p[k]]
        failures += [
            f"preparation: {k} = {p[k]}"
            for k in ("forbidden_feature_columns", "removed_columns_present")
            if p[k]
        ]
        if not p["reproducible_from_raw"]:
            failures.append("preparation: clean.parquet differs from clean(raw); rebuild the data")
        if not p["identifier_columns_present"]:
            failures.append("preparation: identifier columns missing")
    return failures


def validate(cfg: ExperimentConfig) -> dict:
    """Run every check and write ``<validation.out_dir>/report.json``."""
    raw = load_raw(cfg.dataset)
    fresh, clean_report = clean(raw, cfg.dataset)  # schema check: raises on unexpected columns
    report: dict = {
        "raw_csv": str(cfg.dataset.raw_csv),
        "cleaning": clean_report.to_dict(),
        "reported_official_comparison": compare_with_reported(raw),
    }
    if cfg.validation.official_csv is not None:
        official = pd.read_csv(cfg.validation.official_csv, low_memory=False)
        report["official_comparison"] = compare_with_official(raw, official)
    report["preparation"] = check_preparation(raw, fresh, cfg)
    del raw, fresh
    report["prepared_splits"] = check_prepared_splits(cfg)
    report["hard_failures"] = hard_failures(report["prepared_splits"], report["preparation"])
    report["passed"] = not report["hard_failures"]

    out = cfg.validation.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    return report


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
