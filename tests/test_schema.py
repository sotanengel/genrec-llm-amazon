"""Schema validation tests (DESIGN.md §14.2 M0)."""

from __future__ import annotations

import polars as pl
from genrec_lite.data.schema import (
    INTERACTIONS_SCHEMA,
    ITEMS_SCHEMA,
    SAMPLES_SCHEMA,
    USERS_SCHEMA,
    assert_contiguous_ids,
    validate_interactions,
    validate_items,
    validate_samples,
    validate_users,
)


def test_interactions_columns_and_dtypes(mini_bundle: tuple) -> None:
    interactions, _, _, _ = mini_bundle
    validate_interactions(interactions)
    for col, dtype in INTERACTIONS_SCHEMA.items():
        assert interactions.schema[col] == dtype


def test_items_columns_and_dtypes(mini_bundle: tuple) -> None:
    _, items, _, _ = mini_bundle
    validate_items(items)
    for col, dtype in ITEMS_SCHEMA.items():
        assert items.schema[col] == dtype


def test_users_columns_and_dtypes(mini_bundle: tuple) -> None:
    _, _, users, _ = mini_bundle
    validate_users(users)
    for col, dtype in USERS_SCHEMA.items():
        assert users.schema[col] == dtype


def test_samples_columns_and_dtypes(mini_bundle: tuple) -> None:
    _, _, _, samples = mini_bundle
    validate_samples(samples)
    for col, dtype in SAMPLES_SCHEMA.items():
        assert samples.schema[col] == dtype


def test_user_id_is_zero_indexed_contiguous(mini_bundle: tuple) -> None:
    interactions, _, users, _ = mini_bundle
    assert_contiguous_ids(interactions, "user_id")
    assert_contiguous_ids(users, "user_id")


def test_item_id_is_zero_indexed_contiguous(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    assert_contiguous_ids(interactions, "item_id")
    assert_contiguous_ids(items, "item_id")


def test_split_values_in_012(mini_bundle: tuple) -> None:
    interactions, _, _, _ = mini_bundle
    split_values = set(interactions["split"].unique().to_list())
    assert split_values.issubset({0, 1, 2})


def test_no_negative_ts(mini_bundle: tuple) -> None:
    interactions, _, _, _ = mini_bundle
    assert interactions["ts"].min() >= 0


def test_items_first_seen_le_ts(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    item_min_ts = interactions.group_by("item_id").agg(pl.col("ts").min().alias("min_ts"))
    merged = items.join(item_min_ts, on="item_id")
    assert (merged["first_seen_ts"] <= merged["min_ts"]).all()


def test_users_n_inter_matches(mini_bundle: tuple) -> None:
    interactions, _, users, _ = mini_bundle
    inter_counts = interactions.group_by("user_id").len().rename({"len": "n_inter_calc"})
    merged = users.join(inter_counts, on="user_id")
    assert (merged["n_inter"] == merged["n_inter_calc"]).all()
