import pandas as pd
import pytest

from cyberdefense.config import TonIoTConfig
from cyberdefense.data.ton_iot import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    FEATURE_GROUP,
    NUMERIC_COLS,
    TARGET_BINARY,
    TARGET_MULTI,
    clean,
)

CFG = TonIoTConfig()


def test_clean_removes_exact_duplicates(raw_df: pd.DataFrame) -> None:
    df, report = clean(raw_df, CFG)
    assert report.duplicate_rows == 5
    assert report.clean_rows == len(raw_df) - 5 == len(df)
    assert not df.duplicated().any()


def test_rows_differing_only_in_text_columns_are_duplicates(raw_df: pd.DataFrame) -> None:
    twin = raw_df.iloc[[10]].assign(dns_query="another-query")  # differs only in dropped text
    df, report = clean(pd.concat([raw_df, twin], ignore_index=True), CFG)
    assert report.duplicate_rows == 6  # the 5 exact copies + the text-only twin
    assert not df.duplicated().any()


def test_clean_drops_text_columns_and_keeps_ids(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    assert not set(CFG.text_cols) & set(df.columns)
    assert set(CFG.id_cols) <= set(df.columns)
    assert "type" not in df.columns
    assert {TARGET_BINARY, TARGET_MULTI, FEATURE_GROUP} <= set(df.columns)


def test_clean_maps_missing_marker(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    assert all(df[c].dtype == "float32" for c in NUMERIC_COLS)
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in CATEGORICAL_COLS)
    assert (df["http_trans_depth"] == 0).any()  # "-" became 0
    assert "none" in df["proto"].cat.categories  # "-" became "none"
    assert not (df[CATEGORICAL_COLS].astype(str) == "-").any().any()
    assert not df[FEATURE_COLS].isna().any().any()


def test_clean_keeps_exact_duplicate_rows_when_disabled(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG.model_copy(update={"drop_duplicates": False}))
    assert len(df) == len(raw_df)


def test_feature_group_ignores_ids(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    # The synthetic big group differs only in src_port, so it shares one feature_group.
    assert df[FEATURE_GROUP].value_counts().iloc[0] >= 40


@pytest.mark.parametrize("change", ["drop", "add"])
def test_clean_rejects_schema_changes(raw_df: pd.DataFrame, change: str) -> None:
    bad = raw_df.drop(columns="proto") if change == "drop" else raw_df.assign(extra=1)
    with pytest.raises(ValueError, match="columns"):
        clean(bad, CFG)
