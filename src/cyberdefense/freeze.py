"""Freeze the final baselines: a reproducible snapshot later stages compare against.

``freeze`` writes ``<freeze.out_dir>/baselines_<tag>/`` with:

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

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.ton_iot import SPLITS, load_split

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "optuna", "imbalanced-learn", "torch")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprints(cfg: ExperimentConfig) -> dict[str, dict]:
    """Content hash of each split (row order included), independent of parquet metadata."""
    out = {}
    for name in SPLITS:
        df = load_split(cfg.ton_iot, name)
        row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
        out[name] = {
            "rows": len(df),
            "sha256": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
        }
    return out


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


def _collect(cfg: ExperimentConfig) -> dict[str, Path]:
    """Files to snapshot, keyed by their path inside the frozen folder."""
    results = cfg.paths.results_dir
    files = {"data_metadata.json": cfg.ton_iot.processed_dir / "metadata.json"}

    validation = cfg.validation.out_dir / "report.json"
    if not validation.exists():
        raise FileNotFoundError(f"{validation} missing; run scripts/validate_data.py first")
    if not json.loads(validation.read_text())["passed"]:
        raise ValueError(f"{validation} reports failed checks; fix them before freezing")
    files["validation_report.json"] = validation

    baseline_dirs = sorted(p for p in results.glob("baselines*") if (p / "summary.csv").exists())
    if not any(p.name == "baselines" for p in baseline_dirs):
        raise FileNotFoundError(f"{results}/baselines/summary.csv missing; run train_baselines")
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
    dest = cfg.freeze.out_dir / f"baselines_{cfg.freeze.tag}"
    if dest.exists():
        if not cfg.freeze.overwrite:
            raise FileExistsError(
                f"{dest} exists; choose another --freeze.tag or --freeze.overwrite"
            )
        shutil.rmtree(dest)
    files = _collect(cfg)

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
        "raw_csv": {"path": str(cfg.ton_iot.raw_csv), "sha256": file_sha256(cfg.ton_iot.raw_csv)},
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
    """Raise if the current splits differ from the frozen ones."""
    frozen = json.loads(manifest_path.read_text())["splits"]
    current = split_fingerprints(cfg)
    if current != frozen:
        changed = [s for s in SPLITS if current[s] != frozen.get(s)]
        raise ValueError(f"splits differ from {manifest_path}: {changed}; rebuild or re-freeze")
