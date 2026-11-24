import pandas as pd
import pytest

from conftest import CNC_HOST
from cyberdefense.config import DatasetConfig, SplitConfig
from cyberdefense.data.netflow import clean
from cyberdefense.data.schema import DAY, EXCLUDED, FEATURE_GROUP, TARGET_MULTI
from cyberdefense.data.splits import (
    bot_hosts,
    bot_splits,
    check_split,
    clean_attack_days,
    day_split,
    lodo_splits,
    make_split_sets,
    split_groups,
    stratified_group_split,
)

CFG = SplitConfig(max_group_size=20, bot_attack_type="mitm", bot_cnc_hosts=[CNC_HOST])
DAYS = ["2018-02-14", "2018-02-15", "2018-02-16", "2018-02-17"]  # normal, ddos, scanning, mitm


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df, DatasetConfig(normal_window_ratio=1e6, max_attack_flows_per_window=None))[
        0
    ]


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


def test_day_split_holds_out_whole_days(clean_df: pd.DataFrame) -> None:
    split = pd.Series(day_split(clean_df, [DAYS[1]], [DAYS[3]]), index=clean_df.index)
    assert set(clean_df.loc[split == "test", DAY]) == {DAYS[3]}
    assert set(clean_df.loc[split == "val", DAY]) == {DAYS[1]}
    assert not set(clean_df.loc[split == "train", DAY]) & {DAYS[1], DAYS[3]}
    check_split(clean_df, split, CFG, enforce_groups=False)  # reports, does not raise


@pytest.mark.parametrize(
    ("days", "match"),
    [
        (([], []), "test day"),
        (([], ["2030-01-01"]), "not in the data"),
        (([DAYS[1]], [DAYS[1]]), "both"),
    ],
)
def test_day_split_rejects_bad_days(clean_df: pd.DataFrame, days: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        day_split(clean_df, *days)


def test_clean_attack_days_skip_noisy_and_normal_days(clean_df: pd.DataFrame) -> None:
    assert clean_attack_days(clean_df) == DAYS[1:]
    noisy = clean_df.assign(noisy_label=(clean_df[TARGET_MULTI] == "ddos").astype("int8"))
    assert clean_attack_days(noisy) == DAYS[2:]


def test_lodo_holds_out_each_attack_day(clean_df: pd.DataFrame) -> None:
    sets = lodo_splits(clean_df, CFG)
    assert set(sets) == {f"lodo_{d}" for d in DAYS[1:]}
    for day in DAYS[1:]:
        split = pd.Series(sets[f"lodo_{day}"], index=clean_df.index)
        test_types = set(clean_df.loc[split == "test", TARGET_MULTI]) - {"normal"}
        train_types = set(clean_df.loc[split == "train", TARGET_MULTI])
        assert set(clean_df.loc[split == "test", DAY]) == {day}
        assert test_types
        assert not test_types & train_types
        assert DAYS[0] in set(clean_df.loc[split == "train", DAY])  # normal-only day trains
    # val = the next attack day, cyclic
    assert set(clean_df.loc[sets[f"lodo_{DAYS[3]}"] == "val", DAY]) == {DAYS[1]}


def test_lodo_rejects_days_without_clean_attacks(clean_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="without clean attacks"):
        lodo_splits(clean_df, CFG.model_copy(update={"lodo_days": [DAYS[0]]}))


def test_bot_sets_mask_cnc_and_bot_hosts(clean_df: pd.DataFrame) -> None:
    sets = bot_splits(clean_df, CFG)
    hosts = bot_hosts(clean_df, CFG)
    assert hosts == ["192.168.1.1", "192.168.1.2", "192.168.1.3"]

    def touches(h: list[str]) -> pd.Series:
        return clean_df["src_ip"].isin(h) | clean_df["dst_ip"].isin(h)

    for name, masked in (("bot", [CNC_HOST]), ("bot_unseen_hosts", [CNC_HOST, *hosts])):
        split = pd.Series(sets[name], index=clean_df.index)
        assert set(clean_df.loc[split == "test", DAY]) == {DAYS[3]}
        assert set(clean_df.loc[split == "val", DAY]) == {DAYS[2]}  # last attack day before
        assert not (touches(masked) & split.isin(["train", "val"])).any()
        assert (split == EXCLUDED).any()
    # The bot hosts were active before the bot day, so bot_unseen_hosts drops more.
    assert (sets["bot_unseen_hosts"] == EXCLUDED).sum() > (sets["bot"] == EXCLUDED).sum()


def test_bot_sets_need_bot_flows(clean_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no 'bot' flows"):
        bot_splits(clean_df, CFG.model_copy(update={"bot_attack_type": "bot"}))


def test_make_split_sets_columns(clean_df: pd.DataFrame) -> None:
    sets = make_split_sets(clean_df, CFG)
    assert list(sets.columns) == [
        "random",
        *(f"lodo_{d}" for d in DAYS[1:]),
        "bot",
        "bot_unseen_hosts",
    ]
    assert len(sets) == len(clean_df)
    assert not (sets["random"] == EXCLUDED).any()
