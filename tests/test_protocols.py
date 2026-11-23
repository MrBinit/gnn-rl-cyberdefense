"""Unseen-host protocol, experiment naming, tuned-for flags and report tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyberdefense.baselines.experiment import (
    attacker_hosts,
    experiment_name,
    holdout_mask,
    iter_specs,
    run_all,
)
from cyberdefense.baselines.tuning import apply_tuned, check_objective_unchanged
from cyberdefense.config import (
    BaselineConfig,
    ExperimentConfig,
    RandomForestConfig,
    TuningConfig,
    XGBoostConfig,
)
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.ton_iot import SPLITS, load_split
from cyberdefense.reporting import report_tables


@pytest.fixture
def splits(tmp_cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    cfg = tmp_cfg.model_copy(
        update={"ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    return {name: load_split(cfg.ton_iot, name) for name in SPLITS}


def test_holdout_mask_by_host_covers_both_directions() -> None:
    df = pd.DataFrame(
        {
            "src_ip": ["a", "b", "c", "b"],
            "dst_ip": ["b", "c", "a", "d"],
            "attack_type": ["x", "y", "x", "y"],
        }
    )
    assert holdout_mask(df, "unseen_host", "b").tolist() == [True, True, False, True]
    assert holdout_mask(df, "unseen_attack", "x").tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="no holdout"):
        holdout_mask(df, "random", "x")


def test_attacker_hosts_threshold(splits: dict[str, pd.DataFrame]) -> None:
    all_attacks = pd.concat(splits.values())
    counts = all_attacks.loc[all_attacks["label"] == 1, "src_ip"].value_counts()
    hosts = attacker_hosts(splits, min_attack_flows=int(counts.iloc[2]))
    assert hosts == counts.index[: len(hosts)].tolist()  # busiest first
    assert len(hosts) >= 3
    assert attacker_hosts(splits, min_attack_flows=10**9) == []


def test_experiment_name_per_tuning_folder() -> None:
    cfg = ExperimentConfig()
    assert experiment_name(cfg) == "baselines"

    def tuned(out_dir: str) -> ExperimentConfig:
        return cfg.model_copy(
            update={
                "baseline": cfg.baseline.model_copy(update={"use_tuned": True}),
                "tuning": TuningConfig(out_dir=Path(out_dir)),
            }
        )

    assert experiment_name(tuned("results/tuning")) == "baselines_tuned"
    assert experiment_name(tuned("results/tuning_disjoint")) == "baselines_tuned_disjoint"


def test_tuned_for_flags_only_on_tuned_unseen_attack_runs() -> None:
    base = ExperimentConfig(tuning=TuningConfig(holdouts=["mitm"]))
    tuned = base.model_copy(
        update={"baseline": base.baseline.model_copy(update={"use_tuned": True})}
    )
    assert not any(s.tuned_for for s in iter_specs(base, ["mitm", "dos"], ["1.2.3.4"]))
    specs = iter_specs(tuned, ["mitm", "dos"], ["mitm"])  # a host literally named "mitm"
    flagged = {(s.protocol, s.holdout) for s in specs if s.tuned_for}
    assert flagged == {("unseen_attack", "mitm")}


def test_unseen_host_run(tmp_cfg: ExperimentConfig, splits: dict[str, pd.DataFrame]) -> None:
    host = attacker_hosts(splits, min_attack_flows=1)[0]
    cfg = tmp_cfg.model_copy(
        update={
            "baseline": BaselineConfig(
                models=["random_forest"],
                protocols=["unseen_host"],
                random_forest=RandomForestConfig(n_estimators=10),
                unseen_host_min_attack_flows=10**9,  # no hosts selected...
            )
        }
    )
    assert run_all(cfg, splits).empty  # ...so nothing runs

    attacks = pd.concat(splits.values())
    n_flows = int((attacks.loc[attacks["label"] == 1, "src_ip"] == host).sum())
    cfg.baseline.unseen_host_min_attack_flows = n_flows
    summary = run_all(cfg, splits)
    assert host in summary["holdout"].tolist()
    assert (summary["protocol"] == "unseen_host").all()
    assert summary["test_holdout_recall"].between(0, 1).all()
    for col in (
        "test_holdout_recall_at_fpr0p25",
        "test_holdout_recall_at_fpr1",
        "test_holdout_roc_auc",
        "val_fpr_at_fpr1",
    ):
        assert summary[col].between(0, 1).all(), col
    assert (summary["val_fpr_at_fpr1"] <= 0.01).all()  # chosen on val, so capped there

    run_file = next((cfg.paths.results_dir / "baselines" / "unseen_host").rglob("*.json"))
    result = json.loads(run_file.read_text())
    held = holdout_mask(splits["train"], "unseen_host", result["holdout"])
    assert result["rows"]["train"] == len(splits["train"]) - int(held.sum())
    assert result["holdout_test_attack_flows"] > 0
    assert set(result["fixed_fpr_thresholds"]) == {"fpr0p25", "fpr1"}


def test_report_tables_mark_tuned_for_and_aggregate_seeds() -> None:
    rows = []
    for seed, bump in ((1, 0.0), (2, 0.02)):
        for holdout, tuned_for in (("mitm", True), ("dos", False)):
            rows.append(
                {
                    "protocol": "unseen_attack",
                    "task": "binary",
                    "model": "random_forest",
                    "imbalance": "class_weight",
                    "holdout": holdout,
                    "tuned_for": tuned_for,
                    "seed": seed,
                    "test_holdout_recall": 0.9 + bump,
                    "test_holdout_recall_at_fpr1": 0.8 + bump,
                    "test_holdout_roc_auc": 0.99,
                    "test_fpr": 0.01,
                }
            )
    tables = report_tables(pd.DataFrame(rows))
    recall = tables["unseen_attack: recall on the held-out type"]
    assert "mitm †" in recall.index
    assert "dos" in recall.index
    assert recall.loc["dos", "random_forest"] == f"0.9100 ± {np.std([0.9, 0.92], ddof=1):.4f}"
    summary = tables["unseen_attack: summary"]
    assert {"genuine_unseen_recall", "tuned_for_recall"} <= set(summary.columns)
    # Same-FPR summary covers genuinely unseen holdouts only (dos here, not mitm †).
    assert summary.loc["random_forest", "mean_recall_at_fpr1"].startswith("0.8100")
    assert "mean_holdout_roc_auc" in summary.columns
    at_fpr = tables["unseen_attack: recall on the held-out type at 1% FPR (val-chosen)"]
    assert at_fpr.loc["mitm †", "random_forest"].startswith("0.8100")


def test_tuning_folder_guard(tmp_path: Path) -> None:
    model_dir = tmp_path / "xgboost"
    model_dir.mkdir()
    saved = {"tuning_config": json.loads(TuningConfig(holdouts=["mitm"]).model_dump_json())}
    (model_dir / "best_params.json").write_text(json.dumps(saved))

    same = ExperimentConfig(tuning=TuningConfig(holdouts=["mitm"], xgb_trials=99))
    check_objective_unchanged(same, model_dir)  # budget changes are fine
    other = ExperimentConfig(tuning=TuningConfig(holdouts=["dos"]))
    with pytest.raises(ValueError, match="holdouts"):
        check_objective_unchanged(other, model_dir)


def test_apply_tuned_takes_holdouts_from_saved_results(tmp_path: Path) -> None:
    for model in ("random_forest", "xgboost"):
        (tmp_path / model).mkdir()
        payload = {
            "params": {"max_depth": 5},
            "tuning_config": json.loads(TuningConfig(holdouts=["mitm", "dos"]).model_dump_json()),
        }
        (tmp_path / model / "best_params.json").write_text(json.dumps(payload))
    cfg = ExperimentConfig(
        baseline=BaselineConfig(
            use_tuned=True, random_forest=RandomForestConfig(), xgboost=XGBoostConfig()
        ),
        tuning=TuningConfig(out_dir=tmp_path, holdouts=["xss"]),  # CLI says xss...
    )
    tuned = apply_tuned(cfg)
    assert tuned.tuning.holdouts == ["dos", "mitm"]  # ...but the saved results win
    assert tuned.baseline.random_forest.max_depth == 5
    assert tuned.baseline.xgboost.max_depth == 5
