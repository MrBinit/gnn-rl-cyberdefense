import numpy as np
import pandas as pd

from cyberdefense.config import DatasetConfig
from cyberdefense.data.netflow import clean, numeric_cols
from cyberdefense.data.preprocess import fit_transform_splits

CFG = DatasetConfig(min_category_freq=2, normal_window_ratio=1e6, max_attack_flows_per_window=None)


def _halves(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = clean(raw_df, CFG)[0].sample(frac=1, random_state=0)
    return df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]


def test_output_is_finite_float32(raw_df: pd.DataFrame) -> None:
    train, test = _halves(raw_df)
    _, (x_train, x_test) = fit_transform_splits(CFG, train, test)
    assert x_train.dtype == x_test.dtype == np.float32
    assert x_train.shape[1] == x_test.shape[1]
    assert np.isfinite(x_train).all()
    assert np.isfinite(x_test).all()


def test_scaler_is_fitted_on_train_only(raw_df: pd.DataFrame) -> None:
    train, test = _halves(raw_df)
    _, (x_train, x_test) = fit_transform_splits(CFG, train, test)
    n_num = len(numeric_cols(CFG))
    np.testing.assert_allclose(x_train[:, :n_num].mean(axis=0), 0, atol=1e-4)
    # Test statistics are not re-centred, which would leak test information.
    assert not np.allclose(x_test[:, :n_num].mean(axis=0), 0, atol=1e-4)


def test_unseen_category_does_not_fail(raw_df: pd.DataFrame) -> None:
    train, test = _halves(raw_df)
    test = test.copy()
    test["PROTOCOL"] = test["PROTOCOL"].cat.add_categories("never-seen")
    test.loc[test.index[0], "PROTOCOL"] = "never-seen"
    _, (x_train, x_test) = fit_transform_splits(CFG, train, test)
    assert x_test.shape[1] == x_train.shape[1]


def test_log_transform_toggle(raw_df: pd.DataFrame) -> None:
    train, _ = _halves(raw_df)
    pre_log, _ = fit_transform_splits(CFG, train)
    pre_raw, _ = fit_transform_splits(CFG.model_copy(update={"log_transform": False}), train)
    assert len(pre_log.named_transformers_["num"].steps) == 2
    assert len(pre_raw.named_transformers_["num"].steps) == 1
