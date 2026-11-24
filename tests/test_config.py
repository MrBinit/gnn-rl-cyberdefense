"""Per-split-set output locations (ExperimentConfig path properties)."""

from pathlib import Path

from cyberdefense.config import DatasetConfig, ExperimentConfig, GNNConfig, PathsConfig


def _cfg(split_set: str, **kwargs) -> ExperimentConfig:
    return ExperimentConfig(dataset=DatasetConfig(split_set=split_set), **kwargs)


def test_every_result_path_is_inside_the_split_set_folder() -> None:
    cfg = _cfg("bot_unseen_hosts")
    root = Path("results/bot_unseen_hosts")
    assert cfg.run_dir == root
    assert cfg.run_tb_dir == root / "tensorboard"
    assert cfg.tuning_dir == root / "tuning_disjoint"
    assert cfg.graph_report_dir == root / "graph"
    assert cfg.graph_dir == Path("data/processed/graph/bot_unseen_hosts")
    assert cfg.frozen_baselines_dir == Path("frozen/baselines_nf_cicids2018_v1/bot_unseen_hosts")


def test_split_sets_never_share_a_folder() -> None:
    a, b = _cfg("random"), _cfg("lodo_2018-02-14")
    for prop in ("run_dir", "run_tb_dir", "tuning_dir", "graph_dir", "frozen_baselines_dir"):
        assert getattr(a, prop) != getattr(b, prop), prop


def test_absolute_paths_are_used_as_given(tmp_path: Path) -> None:
    cfg = _cfg("bot", paths=PathsConfig(results_dir=tmp_path, tb_dir=tmp_path / "tb"))
    assert cfg.run_dir == tmp_path / "bot"
    assert cfg.run_tb_dir == tmp_path / "tb"


def test_baseline_summaries_default_to_the_frozen_split_set(tmp_path: Path) -> None:
    cfg = _cfg("bot")
    cfg.freeze.out_dir = tmp_path
    summary = cfg.frozen_baselines_dir / "results" / "baselines" / "summary.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text("model\n")
    assert cfg.baseline_summaries == [summary]
    explicit = _cfg("bot", gnn=GNNConfig(compare_with=[Path("x.csv")]))
    assert explicit.baseline_summaries == [Path("x.csv")]
