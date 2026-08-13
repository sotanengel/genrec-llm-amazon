"""Amazon loader unit tests (no HF download)."""

from __future__ import annotations

from genrec_lite.data.loaders.amazon import (
    DESCRIPTION_MAX_LEN,
    build_interactions_from_records,
    filter_5core,
    normalize_timestamp,
    prepare_from_records,
    truncate_description,
)
from genrec_lite.data.stats import compute_stats


def test_normalize_timestamp_milliseconds() -> None:
    assert normalize_timestamp(1_600_000_000_000) == 1_600_000_000


def test_normalize_timestamp_seconds() -> None:
    assert normalize_timestamp(1_600_000_000) == 1_600_000_000


def test_truncate_description() -> None:
    text = "a" * 600
    result = truncate_description(text, max_len=DESCRIPTION_MAX_LEN)
    assert len(result) == DESCRIPTION_MAX_LEN


def test_filter_5core() -> None:
    records = [
        {"user_id": "u0", "parent_asin": "i0", "timestamp": 1_600_000_000_000, "rating": 5.0},
        {"user_id": "u0", "parent_asin": "i1", "timestamp": 1_600_000_086_400, "rating": 4.0},
        {"user_id": "u0", "parent_asin": "i2", "timestamp": 1_600_000_172_800, "rating": 3.0},
        {"user_id": "u1", "parent_asin": "i0", "timestamp": 1_600_000_000_000, "rating": 5.0},
    ]
    df = build_interactions_from_records(records)
    filtered = filter_5core(df, min_core=3)
    # u1 and some items should be filtered out
    assert filtered.height <= df.height


def test_prepare_from_records_produces_valid_bundle(
    mini_review_records: list,
    mini_meta_records: list,
    tmp_path,
) -> None:
    out_dir = tmp_path / "amazon_test"
    prepare_from_records(
        mini_review_records,
        mini_meta_records,
        out_dir,
        split_strategy="global_temporal",
        min_core=3,
    )
    stats = compute_stats(out_dir)
    assert stats.n_users > 0
    assert stats.n_items > 0
    assert stats.n_interactions > 0
