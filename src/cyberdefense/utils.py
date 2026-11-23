"""Reproducibility and logging helpers shared by every experiment."""

import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds libraries that use the global state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
