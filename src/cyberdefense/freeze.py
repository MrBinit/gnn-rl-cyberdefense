"""Frozen snapshots later stages must match.

``freeze_data`` writes ``<freeze.out_dir>/data_<tag>/``: a manifest with the raw-data hash,
the hashes of ``clean.parquet`` / ``splits.parquet``, a content fingerprint of every split of
every split set, the config, environment and git commit, plus the dataset metadata and the
(passed) validation report. ``verify_splits`` checks the configured split set against it.

``freeze`` writes ``<freeze.out_dir>/baselines_<tag>/<split_set>/`` with:

- ``manifest.json``: raw-data hash, split fingerprints, full config, environment,
  git commit, and a checksum for every copied file;
- the dataset metadata and the (passed) validation report;
- every tuning result (``best_params.json``, ``trials.csv``) and baseline summary.

``verify_splits`` lets a later stage (the GNNs) prove it uses identical splits.
"""

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.schema import SPLIT_SETS_FILE, SPLITS, load_clean, load_split_sets

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "optuna", "imbalanced-learn", "torch")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints(row_hashes: np.ndarray, split: np.ndarray) -> dict[str, dict]:
    return {
        name: {
            "rows": int((split == name).sum()),
            "sha256": hashlib.sha256(row_hashes[split == name].tobytes()).hexdigest(),
        }
        for name in SPLITS
    }


def all_split_fingerprints(cfg: ExperimentConfig) -> dict[str, dict[str, dict]]:
    """Content hash of every split of every split set (row order included)."""
    row_hashes = pd.util.hash_pandas_object(load_clean(cfg.dataset), index=False).to_numpy()
    sets = load_split_sets(cfg.dataset)
    return {name: _fingerprints(row_hashes, sets[name].astype(str).to_numpy()) for name in sets}


def split_fingerprints(cfg: ExperimentConfig) -> dict[str, dict]:
    """Fingerprints of the configured split set (``dataset.split_set``)."""
    return all_split_fingerprints(cfg)[cfg.dataset.split_set]


def environment() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def _passed_validation(cfg: ExperimentConfig) -> Path:
    validation = cfg.validation.out_dir / "report.json"
    if not validation.exists():
        raise FileNotFoundError(f"{validation} missing; run scripts/validate_data.py first")
    if not json.loads(validation.read_text())["passed"]:
        raise ValueError(f"{validation} reports failed checks; fix them before freezing")
    return validation


def _prepare_dest(dest: Path, overwrite: bool) -> None:
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dest} exists; choose another --freeze.tag or --freeze.overwrite"
            )
        shutil.rmtree(dest)


def freeze_data(cfg: ExperimentConfig) -> Path:
    """Freeze the prepared data and every split set (after validate_data passed)."""
    dest = cfg.freeze.out_dir / f"data_{cfg.freeze.tag}"
    validation = _passed_validation(cfg)
    _prepare_dest(dest, cfg.freeze.overwrite)
    dest.mkdir(parents=True)
    files = {
        "data_metadata.json": cfg.dataset.processed_dir / "metadata.json",
        "validation_report.json": validation,
    }
    for rel, src in files.items():
        shutil.copy2(src, dest / rel)
    processed = cfg.dataset.processed_dir
    manifest = {
        "tag": cfg.freeze.tag,
        "kind": "data",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_csv": {"path": str(cfg.dataset.raw_csv), "sha256": file_sha256(cfg.dataset.raw_csv)},
        "prepared_files": {
            name: file_sha256(processed / name) for name in ("clean.parquet", SPLIT_SETS_FILE)
        },
        "split_sets": all_split_fingerprints(cfg),
        "config": {
            "dataset": json.loads(cfg.dataset.model_dump_json()),
            "split": json.loads(cfg.split.model_dump_json()),
        },
        "environment": environment(),
        "git": git_state(),
        "files": {rel: file_sha256(dest / rel) for rel in files},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def _collect(cfg: ExperimentConfig) -> dict[str, Path]:
    """Files to snapshot, keyed by their path inside the frozen folder."""
    results = cfg.run_dir
    files = {"data_metadata.json": cfg.dataset.processed_dir / "metadata.json"}
    files["validation_report.json"] = _passed_validation(cfg)

    baseline_dirs = sorted(p for p in results.glob("baselines*") if (p / "summary.csv").exists())
    if not any(p.name == "baselines" for p in baseline_dirs):
        raise FileNotFoundError(
            f"{results}/baselines/summary.csv missing; run train_baselines for this split set"
        )
    for d in baseline_dirs:
        for name in ("summary.csv", "summary_aggregated.csv"):
            if (d / name).exists():
                files[f"results/{d.name}/{name}"] = d / name
    for f in sorted(results.glob("tuning*/*/best_params.json")):
        for name in ("best_params.json", "trials.csv"):
            if (f.parent / name).exists():
                files[f"results/{f.parent.parent.name}/{f.parent.name}/{name}"] = f.parent / name
    return files


def freeze(cfg: ExperimentConfig) -> Path:
    """Write the snapshot; refuses to overwrite an existing tag unless configured to."""
    dest = cfg.frozen_baselines_dir
    files = _collect(cfg)
    _prepare_dest(dest, cfg.freeze.overwrite)

    for rel, src in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    seeds = {}
    for rel in files:
        if rel.endswith("/summary.csv"):
            seeds[rel.split("/")[1]] = sorted(pd.read_csv(dest / rel)["seed"].unique().tolist())

    manifest = {
        "tag": cfg.freeze.tag,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_csv": {"path": str(cfg.dataset.raw_csv), "sha256": file_sha256(cfg.dataset.raw_csv)},
        "split_set": cfg.dataset.split_set,
        "splits": split_fingerprints(cfg),
        "seeds_per_experiment": seeds,
        "config": json.loads(cfg.model_dump_json()),
        "environment": environment(),
        "git": git_state(),
        "files": {rel: file_sha256(dest / rel) for rel in files},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def verify_splits(cfg: ExperimentConfig, manifest_path: Path) -> None:
    """Raise if the configured split set differs from the frozen one.

    Works with a data manifest (``split_sets``: every set) and a baseline manifest
    (``splits`` of its ``split_set``).
    """
    manifest = json.loads(manifest_path.read_text())
    name = cfg.dataset.split_set
    if "split_sets" in manifest:
        if name not in manifest["split_sets"]:
            raise ValueError(f"split set {name!r} not frozen in {manifest_path}")
        frozen = manifest["split_sets"][name]
    else:
        if manifest.get("split_set", "random") != name:
            raise ValueError(f"{manifest_path} froze split set {manifest.get('split_set')!r}")
        frozen = manifest["splits"]
    current = split_fingerprints(cfg)
    if current != frozen:
        changed = [s for s in SPLITS if current[s] != frozen.get(s)]
        raise ValueError(
            f"split set {name!r} differs from {manifest_path}: {changed}; rebuild or re-freeze"
        )
