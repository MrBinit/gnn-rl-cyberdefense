"""Feature preprocessing, fitted on the training split only.

The same fitted preprocessor produces the inputs for every model (RF, XGBoost,
GNN edge features, RL observations), so all comparisons see identical features.
Identifiers (IPs, ports) and time are never features (``cyberdefense.data.netflow``).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cyberdefense.config import DatasetConfig
from cyberdefense.data.netflow import CATEGORICAL_COLS, numeric_cols


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(cfg: DatasetConfig) -> ColumnTransformer:
    numeric_steps = [StandardScaler()]
    if cfg.log_transform:
        # Byte/packet counts and throughputs span many orders of magnitude; log1p makes them
        # usable for neural nets. Clip at 0 first: a few NetFlow fields can be negative.
        numeric_steps.insert(0, FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one"))
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=cfg.min_category_freq,
        sparse_output=False,
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("num", make_pipeline(*numeric_steps), numeric_cols(cfg)),
            ("cat", categorical, CATEGORICAL_COLS),
        ]
    )


def fit_transform_splits(
    cfg: DatasetConfig, train: pd.DataFrame, *others: pd.DataFrame
) -> tuple[ColumnTransformer, list[np.ndarray]]:
    """Fit on ``train``, then transform train and every other split with that fit."""
    pre = make_preprocessor(cfg)
    matrices = [pre.fit_transform(train).astype(np.float32)]
    matrices += [pre.transform(df).astype(np.float32) for df in others]
    return pre, matrices
