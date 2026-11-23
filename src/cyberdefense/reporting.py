"""Result tables shared by every model family (baselines, GNNs, later RL).

Pandas and matplotlib only, no xgboost or torch, so any stage can import it (see the
OpenMP rule in ``cyberdefense/__init__``). Every stage writes per-run results with the
same keys, so one set of tables compares all of them like for like.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from cyberdefense.data.protocols import holdout_label

GROUP_KEYS = ["protocol", "task", "model", "imbalance", "holdout", "tuned_for"]


def confusion_figure(matrix: list[list[int]], labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.8 + 0.7 * len(labels)))
    counts = np.asarray(matrix)
    # Row-normalised: each row shows where that true class ends up (recall per class).
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ConfusionMatrixDisplay(rates, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False, xticks_rotation=45
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def summarize(results: list[dict]) -> pd.DataFrame:
    """One row per run: identifiers, timings and every val/test metric."""
    rows = []
    for r in results:
        row = {k: r[k] for k in (*GROUP_KEYS, "seed", "threshold", "fit_seconds")}
        row |= {f"val_{k}": v for k, v in r["val"].items()}
        row |= {f"test_{k}": v for k, v in r["test"].items()}
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds for every metric column."""
    metric_cols = [c for c in summary.columns if c.startswith(("val_", "test_"))]
    # Older summaries predate the tuned_for column; treat them as not tuned for.
    summary = summary.assign(tuned_for=summary.get("tuned_for", False))
    grouped = summary.fillna({"holdout": "-"}).groupby(GROUP_KEYS, sort=False)[metric_cols]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg.insert(0, "n_seeds", grouped.size())
    return agg.reset_index()


def _fmt(mean: float, std: float, n: int) -> str:
    """'0.9466' for one seed, '0.9466 ± 0.0012' for several."""
    if pd.isna(mean):
        return "-"
    return f"{mean:.4f}" if n <= 1 or pd.isna(std) else f"{mean:.4f} ± {std:.4f}"


def _label(df: pd.DataFrame) -> pd.Series:
    """Column label per run: the model, plus the imbalance strategy when several are compared."""
    if (df["imbalance"] != df["imbalance"].iloc[0]).any():
        return df["model"] + "/" + df["imbalance"]
    return df["model"]


def _mean_std_table(df: pd.DataFrame, index: str | list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean ± std over seeds of each metric, one row per ``index`` value."""
    grouped = df.groupby(index, sort=False)
    stats = grouped[metrics].agg(["mean", "std"])
    n = grouped["seed"].nunique()
    return pd.DataFrame(
        {
            m.removeprefix("test_"): [
                _fmt(stats.loc[i, (m, "mean")], stats.loc[i, (m, "std")], n.loc[i])
                for i in stats.index
            ]
            for m in metrics
        },
        index=stats.index,
    )


def report_tables(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Readable result tables (test metrics; mean ± std when several seeds ran)."""
    summary = summary.assign(
        label=_label(summary), tuned_for=summary.get("tuned_for", False)
    ).fillna({"holdout": "-"})
    tables: dict[str, pd.DataFrame] = {}

    fixed_cols = sorted(c for c in summary.columns if c.startswith("test_recall_at_fpr"))
    random_runs = summary[summary["protocol"] == "random"]
    for task, cols in (
        (
            "binary",
            ["test_f1", "test_precision", "test_recall", "test_fpr", "test_mcc", *fixed_cols],
        ),
        ("multiclass", ["test_f1", "test_balanced_accuracy", "test_mcc", "test_roc_auc"]),
    ):
        rows = random_runs[random_runs["task"] == task]
        if not rows.empty:
            tables[f"random split, {task}"] = _mean_std_table(rows, "label", cols)

    for protocol, what in (
        ("unseen_attack", "type"),
        ("unseen_host", "host"),
        ("unseen_pair", "host-pair fold"),
        ("unseen_attack_host", "type from unseen hosts"),
    ):
        runs = summary[summary["protocol"] == protocol]
        if runs.empty:
            continue
        label = runs["holdout"].map(holdout_label)
        runs = runs.assign(holdout=np.where(runs["tuned_for"].astype(bool), label + " †", label))
        fixed = sorted(c for c in runs.columns if c.startswith("test_holdout_recall_at_"))
        matched = sorted(c for c in runs.columns if c.startswith("test_holdout_tpr_at_"))
        for col in ["test_holdout_recall", *fixed, *matched]:
            if col.startswith("test_holdout_tpr_at_"):
                at = f" at matched {_pct(col)} actual test FPR"
            elif col != "test_holdout_recall":
                at = f" at {_pct(col)} FPR (val-chosen)"
            else:
                at = ""
            per_holdout = runs.groupby(["holdout", "label"], sort=False)[col]
            stats = per_holdout.agg(["mean", "std", "count"])
            stats["cell"] = stats.apply(lambda r: _fmt(r["mean"], r["std"], r["count"]), axis=1)
            tables[f"{protocol}: recall on the held-out {what}{at}"] = (
                stats.reset_index().pivot_table(
                    index="holdout", columns="label", values="cell", aggfunc="first", sort=False
                )
            )

        # Average over holdouts within each seed first, then mean ± std across seeds.
        per_seed = runs.groupby(["label", "seed"]).agg(
            mean_recall=("test_holdout_recall", "mean"),
            worst_recall=("test_holdout_recall", "min"),
            mean_fpr=("test_fpr", "mean"),
            worst_fpr=("test_fpr", "max"),
        )
        genuine_runs = runs[~runs["tuned_for"].astype(bool)]
        genuine = genuine_runs.groupby(["label", "seed"])
        if runs["tuned_for"].any():
            tuned = runs[runs["tuned_for"].astype(bool)].groupby(["label", "seed"])
            per_seed["genuine_unseen_recall"] = genuine["test_holdout_recall"].mean()
            per_seed["tuned_for_recall"] = tuned["test_holdout_recall"].mean()
        # Same-FPR comparison (thresholds from val): over genuinely unseen holdouts only.
        for col in sorted(c for c in runs.columns if c.startswith("test_holdout_recall_at_")):
            name = col.removeprefix("test_holdout_")
            per_seed[f"mean_{name}"] = genuine[col].mean()
            per_seed[f"worst_{name}"] = genuine[col].min()
        if "test_holdout_roc_auc" in runs.columns:
            per_seed["mean_holdout_roc_auc"] = genuine["test_holdout_roc_auc"].mean()
        tables[f"{protocol}: summary"] = _mean_std_table(
            per_seed.reset_index(), "label", [c for c in per_seed.columns]
        )
    return tables


def label_runs(summary: pd.DataFrame, source: str) -> pd.DataFrame:
    """Tag a summary's model names with their source so several summaries can be mixed.

    ``source`` is the results folder name: 'baselines' and 'gnn' keep the plain model name;
    any other ('baselines_tuned_disjoint', 'gnn_transductive') is appended in brackets.
    """
    plain = source in ("baselines", "gnn")
    suffix = "" if plain else f" [{source.removeprefix('baselines_').removeprefix('gnn_')}]"
    return summary.assign(model=summary["model"].astype(str) + suffix)


def headline_table(summary: pd.DataFrame, tuning_holdouts: list[str]) -> pd.DataFrame:
    """One row per model, the comparison docs/03 prescribes (test split, mean ± std over seeds).

    - random binary: F1 and recall at each fixed FPR (thresholds chosen on val);
    - random multiclass: macro-F1;
    - unseen_attack: mean and worst recall at each fixed FPR over the **genuinely unseen**
      types (not in ``tuning_holdouts``, for every model alike), plus mean holdout ROC-AUC;
    - unseen_host: mean and worst recall at each fixed FPR over the held-out hosts.

    Holdout means are taken within each seed first, then mean ± std across seeds.
    """
    summary = summary.assign(label=_label(summary)).fillna({"holdout": "-"})
    blocks = []

    def block(rows: pd.DataFrame, cols: dict[str, str]) -> None:
        if rows.empty:
            return
        renamed = rows.rename(columns=cols)
        blocks.append(_mean_std_table(renamed, "label", [c for c in cols.values() if c in renamed]))

    random_runs = summary[summary["protocol"] == "random"]
    fixed = sorted(c for c in summary.columns if c.startswith("test_recall_at_fpr"))
    binary = random_runs[random_runs["task"] == "binary"]
    block(
        binary,
        {
            "test_f1": "random F1",
            **{
                k: v
                for c in fixed
                for k, v in (
                    (c, f"random {_at(c)}"),
                    (c.replace("recall_at_", "fpr_at_"), f"random actual test FPR@{_pct(c)}"),
                )
            },
        },
    )
    multi = random_runs[random_runs["task"] == "multiclass"]
    block(multi, {"test_f1": "multiclass macro-F1"})

    held = sorted(c for c in summary.columns if c.startswith("test_holdout_recall_at_fpr"))
    matched = sorted(c for c in summary.columns if c.startswith("test_holdout_tpr_at_fpr"))
    for protocol, name in (
        ("unseen_attack", "unseen attack"),
        ("unseen_host", "unseen host"),
        ("unseen_pair", "unseen pair"),
        ("unseen_attack_host", "unseen attack+host"),
    ):
        runs = summary[summary["protocol"] == protocol]
        if protocol == "unseen_attack":
            runs = runs[~runs["holdout"].isin(tuning_holdouts)]
        if runs.empty:
            continue
        per_seed = runs.groupby(["label", "seed"])
        stats = {}
        for c in held:
            stats[f"{name} mean {_at(c)}"] = per_seed[c].mean()
            stats[f"{name} worst {_at(c)}"] = per_seed[c].min()
            # The FPR actually reached on test at that val-chosen threshold (all normal flows),
            # and on the holdout's own normal flows where it has any (hosts, pairs).
            fpr_col = c.replace("holdout_recall_at_", "fpr_at_")
            if fpr_col in runs:
                stats[f"{name} actual test FPR@{_pct(c)}"] = per_seed[fpr_col].mean()
            own = c.replace("holdout_recall_at_", "holdout_fpr_at_")
            if own in runs and runs[own].notna().any():
                stats[f"{name} held-out normal FPR@{_pct(c)}"] = per_seed[own].mean()
        for c in matched:  # same actual test FPR for every model (threshold on test normals)
            if runs[c].notna().any():
                stats[f"{name} mean recall@matched {_pct(c)} test FPR"] = per_seed[c].mean()
                stats[f"{name} worst recall@matched {_pct(c)} test FPR"] = per_seed[c].min()
        if protocol in ("unseen_attack", "unseen_attack_host") and "test_holdout_roc_auc" in runs:
            stats[f"{name} mean ROC-AUC"] = per_seed["test_holdout_roc_auc"].mean()
        frame = pd.DataFrame(stats).reset_index()
        block(frame, {k: k for k in stats})

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, axis=1).fillna("-")


def _pct(col: str) -> str:
    """'test_holdout_recall_at_fpr0p25' -> '0.25%'."""
    return col.rsplit("_at_fpr", 1)[1].replace("p", ".") + "%"


def _at(col: str) -> str:
    """'test_holdout_recall_at_fpr0p25' -> 'recall@0.25%FPR'."""
    return f"recall@{_pct(col)}FPR"


def load_summaries(paths: list[Path]) -> pd.DataFrame:
    """Concatenate several summary.csv files, model names tagged by folder (``label_runs``)."""
    frames = [label_runs(pd.read_csv(p), Path(p).parent.name) for p in paths]
    return pd.concat(frames, ignore_index=True)


def comparison_markdown(summary: pd.DataFrame, tuning_holdouts: list[str], title: str) -> str:
    """Headline table plus every ``report_tables`` table as one Markdown document."""
    parts = [
        f"# {title}",
        "",
        "Test split, mean ± std over seeds. Fixed-FPR thresholds are chosen on val normal "
        f"traffic. 'Unseen attack' averages the genuinely unseen types (excluding the tuning "
        f"holdouts {', '.join(tuning_holdouts)}); † marks a tuning target.",
        "",
        "## Headline",
        "```",
        headline_table(summary, tuning_holdouts).T.to_string(),
        "```",
    ]
    for name, table in report_tables(summary).items():
        parts += ["", f"## {name}", "```", table.to_string(), "```"]
    return "\n".join(parts) + "\n"
