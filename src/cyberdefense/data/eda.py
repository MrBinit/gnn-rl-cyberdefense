"""Exploration figures and summary statistics for the prepared NetFlow dataset (docs/09)."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from cyberdefense.data.netflow import NUMERIC_COLS, normalise_attack
from cyberdefense.data.schema import FEATURE_GROUP, TARGET_BINARY, TARGET_MULTI, TIMESTAMP

VOLUME_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"]


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_eda(raw_attack: pd.Series, clean: pd.DataFrame, out_dir: Path) -> dict:
    """Write figures to ``out_dir`` and return (and save) summary statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = pd.DataFrame(
        {
            "raw": normalise_attack(raw_attack).value_counts(),
            "kept (deduplicated + sampled)": clean[TARGET_MULTI].value_counts(),
        }
    ).sort_values("raw", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    counts.plot.bar(ax=ax, logy=True)
    ax.set(title="Flows per class (log scale)", ylabel="flows")
    _save(fig, out_dir / "class_balance.png")

    zeros = (clean[NUMERIC_COLS] == 0).mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    zeros.plot.barh(ax=ax)
    ax.set_title("Share of zeros per numeric NetFlow field (0 = not applicable / none)")
    ax.invert_yaxis()
    _save(fig, out_dir / "sparsity.png")

    days = pd.to_datetime(clean[TIMESTAMP], unit="s").dt.date
    per_day = pd.crosstab(days, clean[TARGET_MULTI])
    fig, ax = plt.subplots(figsize=(11, 5))
    per_day.plot.bar(stacked=True, ax=ax, logy=False, width=0.85)
    ax.set(title="Kept flows per day and class (one attack family per day)", ylabel="flows")
    ax.legend(ncol=3, fontsize=7)
    _save(fig, out_dir / "classes_per_day.png")

    by_src = clean.groupby("src_ip").agg(
        flows=(TARGET_BINARY, "size"), attack_frac=(TARGET_BINARY, "mean")
    )
    top = by_src.sort_values("flows", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["flows"], color=plt.cm.coolwarm(top["attack_frac"]))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_title("Top-25 source IPs by flow count (red = attack, blue = normal)")
    _save(fig, out_dir / "src_ip_purity.png")

    degree = pd.concat([clean["src_ip"], clean["dst_ip"]]).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degree.to_numpy(), bins=np.logspace(0, np.log10(degree.max()) + 0.1, 40))
    ax.set(xscale="log", yscale="log", xlabel="flows touching host (degree)", ylabel="hosts")
    ax.set_title(f"Host degree distribution ({len(degree)} hosts)")
    _save(fig, out_dir / "host_degree.png")

    long = clean[[*VOLUME_FEATURES, TARGET_MULTI]].melt(id_vars=TARGET_MULTI)
    long["value"] = np.log1p(long["value"])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=long, x="variable", y="value", hue=TARGET_MULTI, ax=ax, showfliers=False)
    ax.set(ylabel="log1p(value)", title="Flow volume features by class")
    ax.legend(ncol=5, fontsize=7)
    _save(fig, out_dir / "features_by_class.png")

    pairs = clean.groupby(["src_ip", "dst_ip"])[TARGET_BINARY].agg(["size", "mean"])
    group_size = clean.groupby(FEATURE_GROUP).size()
    summary = {
        "hosts": len(degree),
        "src_hosts": int(clean["src_ip"].nunique()),
        "dst_hosts": int(clean["dst_ip"].nunique()),
        "unique_host_pairs": len(pairs),
        "mixed_label_host_pairs": int(((pairs["mean"] > 0) & (pairs["mean"] < 1)).sum()),
        "pure_src_ip_flow_share": float(
            by_src.loc[by_src["attack_frac"].isin([0, 1]), "flows"].sum() / len(clean)
        ),
        "max_host_degree": int(degree.max()),
        "median_host_degree": float(degree.median()),
        "distinct_feature_vectors": len(group_size),
        "rows_repeating_a_feature_vector": int(len(clean) - len(group_size)),
        "largest_feature_group": int(group_size.max()),
        "days": int(per_day.shape[0]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
