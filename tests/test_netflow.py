"""NetFlow v3 loading, cleaning, feature policy and per-class sampling."""

import numpy as np
import pandas as pd
import pytest

from cyberdefense.config import DatasetConfig
from cyberdefense.data.netflow import (
    CATEGORICAL_COLS,
    REMOVED_FEATURES,
    TTL_COLS,
    WINDOW_FEATURES,
    clean,
    feature_cols,
    load_raw,
    normalise_attack,
    numeric_cols,
    window_features,
)
from cyberdefense.data.schema import (
    DAY,
    FEATURE_GROUP,
    ID_COLS,
    NOISY_LABEL,
    TARGET_BINARY,
    TARGET_MULTI,
    TIMESTAMP,
    WINDOW,
)

CFG = DatasetConfig(normal_window_ratio=1e6, max_attack_flows_per_window=None)


def test_clean_removes_exact_duplicates(raw_df: pd.DataFrame) -> None:
    df, report = clean(raw_df, CFG)
    assert report.duplicate_rows == 5
    assert report.clean_rows == len(raw_df) - 5 == len(df)


def test_clean_output_columns_and_types(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    assert set(ID_COLS) <= set(df.columns)
    assert {TIMESTAMP, TARGET_BINARY, TARGET_MULTI, NOISY_LABEL, FEATURE_GROUP} <= set(df.columns)
    assert all(df[c].dtype == "float32" for c in numeric_cols(CFG))
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in CATEGORICAL_COLS)
    assert set(df[TARGET_MULTI].astype(str)) == {"normal", "ddos", "scanning", "mitm"}
    assert (df[TARGET_BINARY] == (df[TARGET_MULTI] != "normal")).all()
    assert df[TIMESTAMP].between(1.5e9, 1.6e9).all()  # epoch seconds in 2018
    assert "6" in df["PROTOCOL"].cat.categories  # integer codes, not '6.0'


def test_identifiers_time_and_ttl_are_never_features() -> None:
    feats = set(feature_cols(CFG))
    assert not feats & {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"}
    assert not feats & {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "DNS_QUERY_ID"}
    assert not feats & set(TTL_COLS)
    assert set(TTL_COLS) <= set(feature_cols(CFG.model_copy(update={"use_ttl_features": True})))


def test_per_class_sampling_caps_and_is_seeded(raw_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(sampling="per_class", max_normal_rows=30, max_rows_per_attack=20)
    df, report = clean(raw_df, cfg)
    counts = df[TARGET_MULTI].value_counts()
    assert counts["normal"] == 30
    assert (counts.drop("normal") == 20).all()
    assert report.rows_per_class_before_sampling["normal"] > 30
    again, _ = clean(raw_df, cfg)
    pd.testing.assert_frame_equal(df, again)
    other, _ = clean(raw_df, cfg.model_copy(update={"sample_seed": 7}))
    assert not df[TIMESTAMP].equals(other[TIMESTAMP])


def test_noisy_label_flag(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG.model_copy(update={"noisy_attack_types": ["mitm"]}))
    assert (df[NOISY_LABEL] == (df[TARGET_MULTI] == "mitm")).all()


def test_feature_group_ignores_ids_and_time(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    # The synthetic big group differs only in the source port: one feature_group.
    assert df[FEATURE_GROUP].value_counts().iloc[0] >= 40


def test_normalise_attack() -> None:
    raw = pd.Series(["Benign", "DDOS_attack-HOIC", " Bot "])
    assert normalise_attack(raw).tolist() == ["normal", "ddos_attack-hoic", "bot"]


def test_load_raw_in_chunks(tmp_cfg) -> None:
    df = load_raw(tmp_cfg.dataset)  # chunk_rows=150 < rows: several chunks
    assert len(df) == len(pd.read_csv(tmp_cfg.dataset.raw_csv))
    assert df["IN_BYTES"].dtype == "float32"
    assert isinstance(df["Attack"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("change", ["drop", "add"])
def test_clean_rejects_schema_changes(raw_df: pd.DataFrame, change: str) -> None:
    bad = raw_df.drop(columns="PROTOCOL") if change == "drop" else raw_df.assign(extra=1)
    with pytest.raises(ValueError, match="columns"):
        clean(bad, CFG)


def test_removed_features_are_gone(raw_df: pd.DataFrame) -> None:
    df, report = clean(raw_df, CFG)
    assert not set(REMOVED_FEATURES) & set(df.columns)
    assert not set(REMOVED_FEATURES) & set(feature_cols(CFG))
    assert report.removed_features == REMOVED_FEATURES


def test_day_window_and_window_features(raw_df: pd.DataFrame) -> None:
    df, _ = clean(raw_df, CFG)
    assert df[DAY].nunique() == 4  # one synthetic class per day
    assert (df[WINDOW] == (df[TIMESTAMP] // 300).astype("int64")).all()
    assert set(WINDOW_FEATURES) <= set(df.columns)
    assert (df["WIN_PAIR_FLOWS"] >= 1).all()
    assert (df["WIN_SRC_DSTS"] <= df["WIN_SRC_FLOWS"]).all()
    assert df["SRC_INTERNAL"].eq(1).all()  # 192.168.x.x senders are private
    # Window features are model inputs only on request.
    assert not set(WINDOW_FEATURES) & set(feature_cols(CFG))
    with_win = CFG.model_copy(update={"use_window_features": True})
    assert set(WINDOW_FEATURES) <= set(feature_cols(with_win))


def test_window_features_counts() -> None:
    win = window_features(
        src=np.array([0, 0, 0, 1, 0]),
        dst=np.array([5, 5, 6, 5, 5]),
        dst_port=np.array([80, 80, 22, 80, 80]),
        window=np.array([1, 1, 1, 1, 2]),
    )
    assert win["WIN_PAIR_FLOWS"].tolist() == [2, 2, 1, 1, 1]
    assert win["WIN_SRC_FLOWS"].tolist() == [3, 3, 3, 1, 1]
    assert win["WIN_SRC_DSTS"].tolist() == [2, 2, 2, 1, 1]
    assert win["WIN_SRC_DST_PORTS"].tolist() == [2, 2, 2, 1, 1]
    assert win["WIN_DST_SRCS"].tolist() == [2, 2, 1, 2, 1]


def test_window_sampling_keeps_whole_windows_and_caps_attacks(raw_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(normal_window_ratio=0.0, max_attack_flows_per_window=1)
    df, report = clean(raw_df, cfg)
    # Only windows with a (non-noisy) attack survive, so the normal-only day is gone ...
    assert "normal" not in set(df[TARGET_MULTI].astype(str))
    assert report.windows_kept == report.attack_windows_kept < report.windows_total
    # ... and at most one attack flow per (window, type).
    assert df.groupby([WINDOW, TARGET_MULTI], observed=True).size().max() == 1
    # Window features were computed before sampling, on the complete traffic.
    full, _ = clean(raw_df, CFG)
    key = ["src_ip", "dst_ip", "src_port", TIMESTAMP]
    merged = df.merge(full, on=key, suffixes=("", "_full"))
    assert (merged["WIN_SRC_FLOWS"] == merged["WIN_SRC_FLOWS_full"]).all()


def test_excluded_days_are_dropped(raw_df: pd.DataFrame) -> None:
    df, report = clean(raw_df, CFG.model_copy(update={"exclude_days": ["2018-02-14"]}))
    assert "2018-02-14" not in set(df[DAY])
    assert report.excluded_day_rows > 0


def test_rows_differing_only_in_dropped_columns_are_duplicates(raw_df: pd.DataFrame) -> None:
    twin = raw_df.iloc[[10]].assign(DNS_QUERY_ID=12345, FLOW_END_MILLISECONDS=1)
    df, report = clean(pd.concat([raw_df, twin], ignore_index=True), CFG)
    assert report.duplicate_rows == 6  # the 5 exact copies + the twin
    assert not pd.util.hash_pandas_object(df, index=False).duplicated().any()
