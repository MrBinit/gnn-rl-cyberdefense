import pandas as pd
import pytest

from cyberdefense.config import SplitConfig, TonIoTConfig
from cyberdefense.data.splits import check_split, split_groups, stratified_group_split
from cyberdefense.data.ton_iot import FEATURE_GROUP, TARGET_MULTI, clean

CFG = SplitConfig(max_group_size=20)


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df, TonIoTConfig())[0]


def test_split_fractions(clean_df: pd.DataFrame) -> None:
    split = stratified_group_split(clean_df, CFG)
    share = split.value_counts(normalize=True)
    for name, target in CFG.fractions.items():
        assert share[name] == pytest.approx(target, abs=0.03)


def test_split_is_stratified(clean_df: pd.DataFrame) -> None:
    split = stratified_group_split(clean_df, CFG)
    report = check_split(clean_df, split, CFG)
    assert report.max_class_share_deviation < 0.03
    assert set(clean_df.loc[split == "test", TARGET_MULTI]) == set(clean_df[TARGET_MULTI])


def test_small_groups_stay_together(clean_df: pd.DataFrame) -> None:
    split = stratified_group_split(clean_df, CFG)
    small = clean_df.groupby(FEATURE_GROUP)[FEATURE_GROUP].transform("size") <= CFG.max_group_size
    per_group = split[small].groupby(clean_df.loc[small, FEATURE_GROUP]).nunique()
    assert (per_group == 1).all()


def test_big_groups_become_singletons(clean_df: pd.DataFrame) -> None:
    groups = split_groups(clean_df, CFG)
    big = clean_df.groupby(FEATURE_GROUP)[FEATURE_GROUP].transform("size") > CFG.max_group_size
    assert big.any()
    assert len(set(groups[big.to_numpy()])) == big.sum()


def test_unlimited_group_size_keeps_every_group(clean_df: pd.DataFrame) -> None:
    cfg = CFG.model_copy(update={"max_group_size": None})
    assert split_groups(clean_df, cfg).max() + 1 == clean_df[FEATURE_GROUP].nunique()


def test_split_is_deterministic(clean_df: pd.DataFrame) -> None:
    first = stratified_group_split(clean_df, CFG)
    assert first.equals(stratified_group_split(clean_df, CFG))
    other = stratified_group_split(clean_df, CFG.model_copy(update={"seed": 7}))
    assert not first.equals(other)


def test_invalid_fold_counts_rejected(clean_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="train"):
        stratified_group_split(clean_df, SplitConfig(n_folds=4, val_folds=2, test_folds=2))
