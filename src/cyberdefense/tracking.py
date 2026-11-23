"""TensorBoard logging shared by all experiments (baselines, GNN, RL).

View with: ``uv run tensorboard --logdir results/tensorboard``
"""

from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def make_writer(tb_dir: Path, experiment: str, run: str, config_json: str) -> SummaryWriter:
    """One directory per run: ``<tb_dir>/<experiment>/<run>_<timestamp>``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(tb_dir / experiment / f"{run}_{stamp}"))
    writer.add_text("config", f"```json\n{config_json}\n```")
    return writer


def log_metrics(writer: SummaryWriter, split: str, metrics: dict[str, float], step: int) -> None:
    """Log as ``<metric>/<split>`` so train/val/test of one metric share a chart group."""
    for name, value in metrics.items():
        writer.add_scalar(f"{name}/{split}", value, step)
