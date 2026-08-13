"""Split leak tests (DESIGN.md §14.2 — most critical M0 tests)."""

from __future__ import annotations

import polars as pl
import pytest
from genrec_lite.data.split import SPLIT_TEST, SPLIT_TRAIN


def test_global_temporal_no_future_leak(mini_interactions_global: pl.DataFrame) -> None:
    train = mini_interactions_global.filter(pl.col("split") == SPLIT_TRAIN)
    test = mini_interactions_global.filter(pl.col("split") == SPLIT_TEST)
    if test.height == 0:
        pytest.skip("No test split in mini fixture")
    assert train["ts"].max() < test["ts"].min()


def test_leave_one_out_last_per_user_in_test(mini_interactions_loo: pl.DataFrame) -> None:
    for user_id in mini_interactions_loo["user_id"].unique().to_list():
        user_df = mini_interactions_loo.filter(pl.col("user_id") == user_id).sort("ts")
        test_rows = user_df.filter(pl.col("split") == SPLIT_TEST)
        assert test_rows.height == 1
        assert test_rows["ts"][0] == user_df["ts"].max()


def test_leave_one_out_valid_is_second_last(mini_interactions_loo: pl.DataFrame) -> None:
    for user_id in mini_interactions_loo["user_id"].unique().to_list():
        user_df = mini_interactions_loo.filter(pl.col("user_id") == user_id).sort("ts")
        if user_df.height < 2:
            continue
        valid_rows = user_df.filter(pl.col("split") == 1)
        assert valid_rows.height == 1
        second_last_ts = user_df["ts"][-2]
        assert valid_rows["ts"][0] == second_last_ts


def test_all_users_have_train(mini_interactions_global: pl.DataFrame) -> None:
    all_users = set(mini_interactions_global["user_id"].unique().to_list())
    train_users = set(
        mini_interactions_global.filter(pl.col("split") == SPLIT_TRAIN)["user_id"]
        .unique()
        .to_list()
    )
    assert all_users == train_users


def test_target_item_id_in_train_vocab_or_flagged_cold(mini_bundle: tuple) -> None:
    interactions, items, _, samples = mini_bundle
    train_items = set(
        interactions.filter(pl.col("split") == SPLIT_TRAIN)["item_id"].unique().to_list()
    )
    item_n_train = dict(
        zip(items["item_id"].to_list(), items["n_train_inter"].to_list(), strict=True)
    )

    for row in samples.iter_rows(named=True):
        target = row["target_item"]
        if target not in train_items:
            assert row["target_is_cold"] is True
            assert item_n_train.get(target, 0) < 5


def test_history_only_from_before_cutoff(mini_bundle: tuple) -> None:
    interactions, _, _, samples = mini_bundle
    for row in samples.iter_rows(named=True):
        user_id = row["user_id"]
        cutoff = row["cutoff_ts"]
        history = row["history"]
        user_hist = interactions.filter(
            (pl.col("user_id") == user_id) & (pl.col("ts") < cutoff)
        ).sort("ts")
        expected = user_hist["item_id"].to_list()
        assert history == expected


def test_build_samples_repeat_flag(mini_bundle: tuple) -> None:
    _, _, _, samples = mini_bundle
    for row in samples.iter_rows(named=True):
        expected_repeat = row["target_item"] in row["history"]
        assert row["is_repeat"] == expected_repeat
