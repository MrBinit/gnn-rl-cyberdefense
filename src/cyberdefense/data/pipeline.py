"""End-to-end dataset build: NetFlow v3 CSV -> clean + window sampling -> split sets -> parquet."""

import json
import logging

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.eda import write_eda
from cyberdefense.data.netflow import RAW_ATTACK, clean, load_raw
from cyberdefense.data.schema import (
    DAY,
    EXCLUDED,
    NOISY_LABEL,
    SPLIT_SETS_FILE,
    SPLITS,
    TARGET_BINARY,
    TARGET_MULTI,
)
from cyberdefense.data.splits import bot_hosts, check_split, make_split_sets

log = logging.getLogger(__name__)


def build_dataset(cfg: ExperimentConfig) -> dict:
    """Build ``clean.parquet``, ``splits.parquet`` and ``metadata.json``; return the metadata."""
    tcfg, scfg = cfg.dataset, cfg.split
    out = tcfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(tcfg)
    raw_attack = raw[RAW_ATTACK]  # all the EDA needs from the raw table
    df, report = clean(raw, tcfg)
    del raw
    log.info(
        "cleaned %s raw rows: %s duplicates removed, %s kept after sampling",
        f"{report.raw_rows:,}",
        f"{report.duplicate_rows:,}",
        f"{report.clean_rows:,}",
    )

    sets = make_split_sets(df, scfg)
    random = pd.Series(sets["random"].astype(str).to_numpy(), index=df.index)
    checks = check_split(df, random, scfg)  # raises on feature-group leakage

    df.to_parquet(out / "clean.parquet", index=False)
    sets.to_parquet(out / SPLIT_SETS_FILE, index=False)
    for name in SPLITS:  # per-split files of the old layout would now be stale
        (out / f"{name}.parquet").unlink(missing_ok=True)

    days = df[DAY]
    metadata = {
        "dataset": tcfg.name,
        "source": str(tcfg.raw_csv),
        "sampling": {
            "strategy": tcfg.sampling,
            "window_seconds": tcfg.window_seconds,
            "normal_window_ratio": tcfg.normal_window_ratio,
            "max_attack_flows_per_window": tcfg.max_attack_flows_per_window,
            "seed": tcfg.sample_seed,
        },
        "noisy_label_rows": int(df[NOISY_LABEL].sum()),
        "days": sorted(days.unique().tolist()),
        "attack_types_per_day": {
            d: sorted(g.astype(str).unique().tolist())
            for d, g in df.loc[df[TARGET_BINARY] == 1, TARGET_MULTI].groupby(days, observed=True)
        },
        "clean": report.to_dict(),
        "split_config": scfg.model_dump(),
        "bot_hosts": bot_hosts(df, scfg),
        "split_sets": {name: describe_split_set(df, sets[name]) for name in sets.columns},
        "random_checks": {
            "target_fractions": scfg.fractions,
            "feature_group_overlap": checks.group_overlap,
            "max_class_share_deviation": round(checks.max_class_share_deviation, 5),
            "test_rows_with_vector_seen_in_train": round(
                checks.test_rows_with_vector_seen_in_train, 4
            ),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=int))
    log.info(
        "saved clean.parquet + %s (%s split sets) to %s", SPLIT_SETS_FILE, len(sets.columns), out
    )

    if tcfg.make_figures:
        metadata["eda"] = write_eda(raw_attack, df, tcfg.eda_dir)
        log.info("saved figures to %s", tcfg.eda_dir)
    return metadata


def describe_split_set(df: pd.DataFrame, split: pd.Series) -> dict:
    """Rows, days, attack share and class counts per split of one split set."""
    split = pd.Series(split.astype(str).to_numpy(), index=df.index)
    counts = pd.crosstab(df[TARGET_MULTI], split).reindex(columns=SPLITS, fill_value=0)
    clean_attack = (df[TARGET_BINARY] == 1) & (df[NOISY_LABEL] == 0)
    return {
        "rows": {s: int((split == s).sum()) for s in SPLITS},
        "excluded_rows": int((split == EXCLUDED).sum()),
        "days": {s: sorted(df.loc[split == s, DAY].unique().tolist()) for s in SPLITS},
        "attack_share": {
            s: round(float(df.loc[split == s, TARGET_BINARY].mean()), 4) for s in SPLITS
        },
        "clean_attack_rows": {s: int((clean_attack & (split == s)).sum()) for s in SPLITS},
        "class_counts": {s: {k: int(v) for k, v in counts[s].items() if v} for s in SPLITS},
    }
