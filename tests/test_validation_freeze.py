import json

import pandas as pd
import pytest

from cyberdefense.baselines.experiment import run_all
from cyberdefense.config import (
    BaselineConfig,
    ExperimentConfig,
    FreezeConfig,
    RandomForestConfig,
    ValidationConfig,
)
from cyberdefense.data import validation
from cyberdefense.data.netflow import load_raw
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.schema import SPLITS, load_split
from cyberdefense.data.validation import (
    compare_with_official,
    compare_with_reported,
    hard_failures,
    validate,
)
from cyberdefense.freeze import freeze, freeze_data, verify_splits


@pytest.fixture
def cfg(tmp_cfg: ExperimentConfig) -> ExperimentConfig:
    root = tmp_cfg.paths.results_dir
    cfg = tmp_cfg.model_copy(
        update={
            "dataset": tmp_cfg.dataset.model_copy(update={"make_figures": False}),
            "validation": ValidationConfig(out_dir=root / "validation"),
            "freeze": FreezeConfig(out_dir=root.parent / "frozen"),
            "baseline": BaselineConfig(
                models=["random_forest"],
                protocols=["random"],
                random_forest=RandomForestConfig(n_estimators=5),
            ),
        }
    )
    build_dataset(cfg)
    return cfg


def test_validate_passes_on_clean_pipeline(cfg: ExperimentConfig) -> None:
    report = validate(cfg)
    assert report["passed"], report["hard_failures"]
    prepared = report["prepared_splits"]
    assert prepared["exact_duplicate_rows_after_cleaning"] == 0
    for s in prepared["split_sets"].values():
        assert not any(s["row_overlap_between_splits"].values())
        assert s["masked_host_rows_in_train_val"] == 0
    assert prepared["split_sets"]["bot_unseen_hosts"]["masked_hosts"] == 4  # C&C + 3 bots
    prep = report["preparation"]
    assert prep["reproducible_from_raw"]
    assert prep["forbidden_feature_columns"] == []
    assert prep["windows_with_incomplete_normal_traffic"] == 0
    assert prep["rows_not_matching_complete_traffic"] == 0
    assert (cfg.validation.out_dir / "report.json").exists()
    assert "official_comparison" not in report


def test_validate_checks_window_sampling(cfg: ExperimentConfig) -> None:
    """Real window sampling (attack cap, some quiet windows) still passes every check."""
    sampled = cfg.model_copy(
        update={
            "dataset": cfg.dataset.model_copy(
                update={"normal_window_ratio": 0.1, "max_attack_flows_per_window": 1}
            )
        }
    )
    build_dataset(sampled)
    report = validate(sampled)
    # (~100 sampled rows are too few for the 1-point random-split class-share tolerance.)
    assert not [f for f in report["hard_failures"] if f.startswith("preparation")]
    assert report["preparation"]["windows_kept"] < report["cleaning"]["windows_total"]
    assert report["preparation"]["max_attack_flows_per_window_type"] == 1


def test_validate_detects_a_tampered_table(cfg: ExperimentConfig) -> None:
    path = cfg.dataset.processed_dir / "clean.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "WIN_PAIR_FLOWS"] += 1000  # window feature no longer matches the traffic
    df.to_parquet(path, index=False)
    failures = validate(cfg)["hard_failures"]
    assert any("reproducible" in f or "clean(raw)" in f for f in failures)
    assert any("rows_not_matching_complete_traffic" in f for f in failures)


def test_compare_with_reported_detects_normal_subsample(monkeypatch) -> None:
    published = {"normal": 300, "ddos_attack-hoic": 20, "bot": 10}
    monkeypatch.setattr(validation, "REPORTED_OFFICIAL_COUNTS", published)
    counts = dict(published, normal=50)  # a normal-class subsample of the published file
    raw = pd.DataFrame(
        {
            "Attack": [
                ("Benign" if c == "normal" else c) for c, n in counts.items() for _ in range(n)
            ]
        }
    )
    result = compare_with_reported(raw)
    assert result["differing_classes"] == ["normal"]
    assert result["attack_counts_match"]
    assert "normal-class subsample" in result["verdict"]


def test_compare_with_official(cfg: ExperimentConfig) -> None:
    raw = load_raw(cfg.dataset)
    official = pd.concat([raw, raw.iloc[:10].assign(L4_SRC_PORT=1)], ignore_index=True)
    result = compare_with_official(raw.iloc[:-3], official)
    assert result["this_copy_is_subset"]
    assert result["share_of_attack_rows_found"] == 1.0
    assert not compare_with_official(raw, official.drop(columns="PROTOCOL"))["same_columns"]


def test_hard_failures() -> None:
    good_set = {
        "rows": {"train": 5, "val": 2, "test": 2},
        "clean_attack_rows": {"train": 1, "val": 1, "test": 1},
        "row_overlap_between_splits": {"train&test": 0},
        "max_class_share_deviation": 0.001,
        "masked_host_rows_in_train_val": 0,
    }
    ok = {"exact_duplicate_rows_after_cleaning": 0, "split_sets": {"random": good_set}}
    assert hard_failures(ok) == []
    bad_random = good_set | {
        "row_overlap_between_splits": {"train&test": 3},
        "max_class_share_deviation": 0.2,
    }
    assert len(hard_failures(ok | {"split_sets": {"random": bad_random}})) == 2
    # Class shares are only enforced on the random set; masked hosts on every set.
    bot = good_set | {"max_class_share_deviation": 0.5, "masked_host_rows_in_train_val": 4}
    assert hard_failures(ok | {"split_sets": {"bot": bot}}) == [
        "bot: 4 masked-host rows in train/val"
    ]
    empty = good_set | {"rows": {"train": 5, "val": 0, "test": 2}}
    assert hard_failures(ok | {"split_sets": {"lodo_x": empty}}) == ["lodo_x: empty splits ['val']"]


def test_freeze_requires_validation(cfg: ExperimentConfig) -> None:
    with pytest.raises(FileNotFoundError, match="validate_data"):
        freeze(cfg)


def test_freeze_snapshot_and_verify(cfg: ExperimentConfig) -> None:
    validate(cfg)
    run_all(cfg, {name: load_split(cfg.dataset, name) for name in SPLITS})
    dest = freeze(cfg)
    assert dest == cfg.freeze.out_dir / "baselines_nf_cicids2018_v1" / "random"

    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["split_set"] == "random"
    assert set(manifest["splits"]) == set(SPLITS)
    assert manifest["seeds_per_experiment"] == {"baselines": [42]}
    assert "results/baselines/summary.csv" in manifest["files"]
    assert manifest["raw_csv"]["sha256"]
    for rel in manifest["files"]:
        assert (dest / rel).exists()

    with pytest.raises(FileExistsError):
        freeze(cfg)  # same tag again
    verify_splits(cfg, dest / "manifest.json")  # unchanged data passes

    _move_one_test_row_to_train(cfg, "random")
    with pytest.raises(ValueError, match="test"):
        verify_splits(cfg, dest / "manifest.json")


def _move_one_test_row_to_train(cfg: ExperimentConfig, split_set: str) -> None:
    path = cfg.dataset.processed_dir / "splits.parquet"
    sets = pd.read_parquet(path)
    sets[split_set] = sets[split_set].astype(str)  # arrow-backed categoricals are read-only
    sets.loc[sets.index[sets[split_set] == "test"][0], split_set] = "train"
    sets.to_parquet(path, index=False)


def test_freeze_data_and_verify_each_split_set(cfg: ExperimentConfig) -> None:
    validate(cfg)
    dest = freeze_data(cfg)
    manifest = json.loads((dest / "manifest.json").read_text())
    names = set(manifest["split_sets"])
    assert {"random", "bot", "bot_unseen_hosts"} <= names
    assert manifest["prepared_files"]["splits.parquet"]
    assert (dest / "validation_report.json").exists()

    def with_set(name: str) -> ExperimentConfig:
        return cfg.model_copy(
            update={"dataset": cfg.dataset.model_copy(update={"split_set": name})}
        )

    for name in names:
        verify_splits(with_set(name), dest / "manifest.json")
    with pytest.raises(FileExistsError):
        freeze_data(cfg)

    _move_one_test_row_to_train(cfg, "bot")
    verify_splits(with_set("random"), dest / "manifest.json")  # other sets unaffected
    with pytest.raises(ValueError, match="'bot' differs"):
        verify_splits(with_set("bot"), dest / "manifest.json")
