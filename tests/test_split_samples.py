"""Train sample generation tests (M3 prep, DESIGN.md §3.2)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from genrec_lite.data.loaders.amazon import prepare_from_records
from genrec_lite.data.schema import read_parquet_bundle, read_train_samples
from genrec_lite.data.split import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    build_samples,
    build_train_samples,
)


@pytest.fixture
def prepared_bundle(tmp_path: Path, mini_review_records: list, mini_meta_records: list) -> Path:
    out_dir = tmp_path / "bundle"
    prepare_from_records(
        mini_review_records,
        mini_meta_records,
        out_dir,
        split_strategy="global_temporal",
        cold_threshold=5,
        min_core=3,
    )
    return out_dir


def test_build_train_samples_includes_train_split(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    train_samples = build_train_samples(interactions, items)
    assert train_samples.height > 0
    assert set(train_samples["split"].unique().to_list()) == {SPLIT_TRAIN}


def test_train_sample_history_only_from_train_before_cutoff(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    train_samples = build_train_samples(interactions, items)
    train_inter = interactions.filter(pl.col("split") == SPLIT_TRAIN)

    for row in train_samples.iter_rows(named=True):
        user_id = row["user_id"]
        cutoff = row["cutoff_ts"]
        history = row["history"]
        allowed = (
            train_inter.filter((pl.col("user_id") == user_id) & (pl.col("ts") < cutoff))
            .sort("ts")["item_id"]
            .to_list()
        )
        assert history == allowed


def test_train_sample_is_repeat_flag(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    train_samples = build_train_samples(interactions, items)
    for row in train_samples.iter_rows(named=True):
        expected = row["target_item"] in row["history"]
        assert row["is_repeat"] == expected


def test_train_sample_empty_history_excluded(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    train_samples = build_train_samples(interactions, items)
    for row in train_samples.iter_rows(named=True):
        assert len(row["history"]) > 0


def test_train_sample_id_unique_and_stable(mini_bundle: tuple) -> None:
    interactions, items, _, _ = mini_bundle
    first = build_train_samples(interactions, items)
    second = build_train_samples(interactions, items)
    assert first["sample_id"].n_unique() == first.height
    assert first.equals(second)


def test_samples_parquet_unchanged_when_train_added(
    tmp_path: Path,
    mini_review_records: list,
    mini_meta_records: list,
) -> None:
    """samples.parquet must stay valid+test only, identical to build_samples()."""
    from genrec_lite.data.loaders.amazon import (
        build_interactions_from_records,
        build_items_from_records,
        filter_5core,
        remap_ids,
    )
    from genrec_lite.data.split import apply_split, build_samples, compute_item_metadata

    interactions = build_interactions_from_records(mini_review_records)
    interactions = filter_5core(interactions, min_core=3)
    interactions, user_raw_map, item_raw_map = remap_ids(interactions)
    interactions = apply_split(interactions, "global_temporal")
    item_meta = compute_item_metadata(interactions)
    items = build_items_from_records(mini_meta_records, item_raw_map, item_meta)
    expected_samples = build_samples(interactions, items, "global_temporal", cold_threshold=5)

    out_dir = tmp_path / "with_train"
    prepare_from_records(
        mini_review_records,
        mini_meta_records,
        out_dir,
        split_strategy="global_temporal",
        cold_threshold=5,
        min_core=3,
    )
    _, _, _, actual_samples = read_parquet_bundle(out_dir)
    assert actual_samples.equals(expected_samples)
    assert actual_samples.filter(pl.col("split") == SPLIT_TRAIN).height == 0


def test_valid_test_sample_counts_leave_one_out(
    tmp_path: Path,
) -> None:
    """Regression: eval sample counts must match legacy build_samples for leave_one_out."""
    records: list[dict] = []
    base_ts = 1_600_000_000_000
    for u in range(5):
        for i in range(6):
            records.append(
                {
                    "user_id": f"user_{u}",
                    "parent_asin": f"item_{i}",
                    "timestamp": base_ts + u * 1_000_000 + i * 86_400_000,
                    "rating": 5.0,
                }
            )
    meta = [{"parent_asin": f"item_{i}", "title": f"Game {i}"} for i in range(6)]
    out_dir = tmp_path / "loo"
    prepare_from_records(records, meta, out_dir, split_strategy="leave_one_out", min_core=1)
    _, _, _, samples = read_parquet_bundle(out_dir)
    assert samples.filter(pl.col("split") == SPLIT_VALID).height == 5
    assert samples.filter(pl.col("split") == SPLIT_TEST).height == 5


def test_prepare_writes_train_samples_parquet(prepared_bundle: Path) -> None:
    train_samples = read_train_samples(prepared_bundle)
    assert train_samples.height > 0
    assert (prepared_bundle / "train_samples.parquet").exists()
