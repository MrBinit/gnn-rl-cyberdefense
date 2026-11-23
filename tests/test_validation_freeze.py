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
from cyberdefense.data.pipeline import build_dataset
from cyberdefense.data.ton_iot import SPLITS, load_raw, load_split
from cyberdefense.data.validation import (
    REPORTED_OFFICIAL_COUNTS,
    compare_with_official,
    compare_with_reported,
    hard_failures,
    validate,
)
from cyberdefense.freeze import freeze, verify_splits


@pytest.fixture
def cfg(tmp_cfg: ExperimentConfig) -> ExperimentConfig:
    root = tmp_cfg.paths.results_dir
    cfg = tmp_cfg.model_copy(
        update={
            "ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False}),
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
    splits = report["prepared_splits"]
    assert splits["exact_duplicate_rows_after_cleaning"] == 0
    assert not any(splits["row_overlap_between_splits"].values())
    assert (cfg.validation.out_dir / "report.json").exists()
    assert "official_comparison" not in report


def test_compare_with_reported_detects_normal_subsample() -> None:
    counts = dict(REPORTED_OFFICIAL_COUNTS, normal=50_000)
    raw = pd.DataFrame({"type": [c for c, n in counts.items() for _ in range(n)]})
    result = compare_with_reported(raw)
    assert result["differing_classes"] == ["normal"]
    assert result["attack_counts_match"]
    assert "normal-class subsample" in result["verdict"]


def test_compare_with_official(cfg: ExperimentConfig) -> None:
    raw = load_raw(cfg.ton_iot)
    official = pd.concat([raw, raw.iloc[:10].assign(src_port=1)], ignore_index=True)
    result = compare_with_official(raw.iloc[:-3], official)
    assert result["this_copy_is_subset"]
    assert result["share_of_attack_rows_found"] == 1.0
    assert not compare_with_official(raw, official.drop(columns="proto"))["same_columns"]


def test_hard_failures() -> None:
    ok = {
        "exact_duplicate_rows_after_cleaning": 0,
        "row_overlap_between_splits": {"train&test": 0},
        "max_class_share_deviation": 0.001,
    }
    assert hard_failures(ok) == []
    bad = ok | {"row_overlap_between_splits": {"train&test": 3}, "max_class_share_deviation": 0.2}
    assert len(hard_failures(bad)) == 2


def test_freeze_requires_validation(cfg: ExperimentConfig) -> None:
    with pytest.raises(FileNotFoundError, match="validate_data"):
        freeze(cfg)


def test_freeze_snapshot_and_verify(cfg: ExperimentConfig) -> None:
    validate(cfg)
    run_all(cfg, {name: load_split(cfg.ton_iot, name) for name in SPLITS})
    dest = freeze(cfg)

    manifest = json.loads((dest / "manifest.json").read_text())
    assert set(manifest["splits"]) == set(SPLITS)
    assert manifest["seeds_per_experiment"] == {"baselines": [42]}
    assert "results/baselines/summary.csv" in manifest["files"]
    assert manifest["raw_csv"]["sha256"]
    for rel in manifest["files"]:
        assert (dest / rel).exists()

    with pytest.raises(FileExistsError):
        freeze(cfg)  # same tag again
    verify_splits(cfg, dest / "manifest.json")  # unchanged data passes

    test_path = cfg.ton_iot.processed_dir / "test.parquet"
    pd.read_parquet(test_path).iloc[1:].to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="test"):
        verify_splits(cfg, dest / "manifest.json")
