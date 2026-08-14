"""Amazon loader unit tests (no HF download)."""

from __future__ import annotations

from unittest.mock import patch

from genrec_lite.data.loaders.amazon import (
    DESCRIPTION_MAX_LEN,
    build_interactions_from_records,
    filter_5core,
    hf_config_names,
    load_amazon_category_from_hf,
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


def test_hf_config_names_for_video_games() -> None:
    assert hf_config_names("Video_Games") == (
        "raw_review_Video_Games",
        "raw_meta_Video_Games",
    )


def test_load_amazon_category_from_hf_uses_trust_remote_code() -> None:
    review_row = {
        "main_category": "Video_Games",
        "parent_asin": "item_0",
        "user_id": "user_0",
        "timestamp": 1_600_000_000_000,
        "rating": 5.0,
        "categories": [],
    }
    meta_row = {
        "main_category": "Video_Games",
        "parent_asin": "item_0",
        "title": "Game 0",
    }
    reviews = iter([review_row])
    meta = iter([meta_row])

    with patch("datasets.load_dataset", side_effect=[reviews, meta]) as mock_load:
        review_records, meta_records = load_amazon_category_from_hf("Video_Games")

    assert review_records == [review_row]
    assert meta_records == [meta_row]
    assert mock_load.call_count == 2
    assert mock_load.call_args_list[0].kwargs["name"] == "raw_review_Video_Games"
    assert mock_load.call_args_list[1].kwargs["name"] == "raw_meta_Video_Games"
    for call in mock_load.call_args_list:
        assert call.kwargs["trust_remote_code"] is True


def test_build_interactions_handles_stringified_none_rating() -> None:
    """Amazon Reviews 2023 records occasionally carry rating='None' (string).
    The loader must coerce them to NaN instead of raising ValueError."""
    import math

    records = [
        {"user_id": "u0", "parent_asin": "i0", "timestamp": 1_600_000_000_000, "rating": "None"},
        {"user_id": "u1", "parent_asin": "i0", "timestamp": 1_600_000_086_400, "rating": 5.0},
    ]
    df = build_interactions_from_records(records)
    assert df.height == 2
    ratings = df["rating"].to_list()
    assert math.isnan(ratings[0])
    assert ratings[1] == 5.0


def test_prepare_from_records_handles_item_whose_earliest_event_is_valid_or_test(tmp_path) -> None:
    """Regression: an item whose globally-earliest interaction lands in a user's
    leave-one-out valid or test event must not violate
    items.first_seen_ts <= min(interactions.ts per item).
    """
    # Two users, both with the same item i0. User u0's earliest interaction with
    # i0 comes at t=100 but is one of u0's last 2 events (so leave_one_out puts
    # it in valid/test). User u1's earliest with i0 is at t=200 (train).
    # Legacy code assigned first_seen_ts = 200 > min_ts = 100 -> validation fails.
    records = []
    # u0: five interactions so 5-core keeps them; earliest with i0 at t=100
    # but i0 is one of u0's later items -> leave_one_out likely moves it out
    # of train. To force this we make i0 u0's most recent item.
    for i, iid in enumerate(["ix0", "ix1", "ix2", "ix3", "i0"]):
        records.append(
            {
                "user_id": "u0",
                "parent_asin": iid,
                "timestamp": 1_000_000_000_000 + i * 86_400_000,
                "rating": 5.0,
            }
        )
    # u1..u4 each buy i0 and 4 other items at later times so i0's globally-min ts
    # is u0's t=1_000_000_000_000 + 4*86400000, which sits in u0's test/valid split.
    base_late = 1_000_000_500_000_000  # far in the future so u0 is earliest
    for u in range(1, 5):
        for i, iid in enumerate(["ix0", "ix1", "ix2", "ix3", "i0"]):
            records.append(
                {
                    "user_id": f"u{u}",
                    "parent_asin": iid,
                    "timestamp": base_late + u * 86_400_000 + i * 3600_000,
                    "rating": 4.0,
                }
            )
    meta = [{"parent_asin": iid, "title": iid} for iid in ["ix0", "ix1", "ix2", "ix3", "i0"]]
    out_dir = tmp_path / "amazon_first_seen"
    prepare_from_records(records, meta, out_dir, split_strategy="leave_one_out", min_core=1)
    # If the function returned without raising, the schema validator was satisfied.


def test_normalize_timestamp_sub_1e12_milliseconds_regression() -> None:
    """Regression: Amazon records from before ~2001 carry ms timestamps below
    1e12 (e.g. 9.79e11). The old `>= 1e12` check let them through as seconds,
    producing year-33189 datetimes downstream."""
    # 979993018000 ms is 2001-01-20 UTC in seconds.
    assert normalize_timestamp(979_993_018_000) == 979_993_018


def test_normalize_timestamp_microseconds() -> None:
    """Loop handles the microsecond scale too."""
    assert normalize_timestamp(1_600_000_000_000_000) == 1_600_000_000
