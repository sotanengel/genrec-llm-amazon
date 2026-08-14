"""Amazon Reviews 2023 loader (DESIGN.md §3.1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from genrec_lite.data.schema import write_parquet_bundle
from genrec_lite.data.split import (
    SplitStrategy,
    apply_split,
    build_samples,
    compute_item_metadata,
    compute_user_metadata,
)

logger = logging.getLogger(__name__)

DESCRIPTION_MAX_LEN = 500
EVENT_TYPE_REVIEW = 3
MIN_CORE = 5


def normalize_timestamp(ts_raw: int | float) -> int:
    """Convert timestamp to UNIX seconds (DESIGN.md §10.10)."""
    ts = int(ts_raw)
    # Millisecond timestamps are >= 1e12
    if ts >= 1_000_000_000_000:
        return ts // 1000
    return ts


def truncate_description(text: str, max_len: int = DESCRIPTION_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len]


def filter_5core(interactions: pl.DataFrame, min_core: int = MIN_CORE) -> pl.DataFrame:
    """Iteratively filter until all users and items have >= min_core interactions."""
    df = interactions
    while True:
        user_counts = df.group_by("user_id").len()
        item_counts = df.group_by("item_id").len()
        valid_users = user_counts.filter(pl.col("len") >= min_core)["user_id"].implode()
        valid_items = item_counts.filter(pl.col("len") >= min_core)["item_id"].implode()
        filtered = df.filter(
            pl.col("user_id").is_in(valid_users) & pl.col("item_id").is_in(valid_items)
        )
        if filtered.height == df.height:
            break
        df = filtered
    return df


def remap_ids(interactions: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str], dict[str, str]]:
    """Remap raw user/item IDs to 0-indexed contiguous integers."""
    user_ids = interactions["user_id"].unique().sort()
    item_ids = interactions["item_id"].unique().sort()

    user_map = {uid: i for i, uid in enumerate(user_ids.to_list())}
    item_map = {iid: i for i, iid in enumerate(item_ids.to_list())}

    remapped = interactions.with_columns(
        pl.col("user_id").replace_strict(user_map).cast(pl.Int32),
        pl.col("item_id").replace_strict(item_map).cast(pl.Int32),
    )
    user_raw = {str(v): k for k, v in user_map.items()}
    item_raw = {str(v): k for k, v in item_map.items()}
    return remapped, user_raw, item_raw


def build_interactions_from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Build raw interactions dataframe from list of review records."""
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        ts = normalize_timestamp(rec["timestamp"])
        rows.append(
            {
                "user_id": rec["user_id"],
                "item_id": rec["parent_asin"],
                "ts": ts,
                "basket_id": i,
                "rating": float(rec.get("rating", float("nan"))),
                "event_type": EVENT_TYPE_REVIEW,
            }
        )
    return pl.DataFrame(rows)


def build_items_from_records(
    records: list[dict[str, Any]],
    item_raw_map: dict[str, str],
    item_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Build items table from metadata records."""
    meta_by_id = {rec["parent_asin"]: rec for rec in records}
    rows: list[dict[str, Any]] = []
    for item_id_str, raw_id in item_raw_map.items():
        item_id = int(item_id_str)
        meta = meta_by_id.get(raw_id, {})
        title = str(meta.get("title", ""))
        brand = str(meta.get("store", meta.get("brand", "")))
        categories = meta.get("categories", [])
        if isinstance(categories, list) and categories:
            if isinstance(categories[0], list):
                category_path = " > ".join(categories[0])
            else:
                category_path = " > ".join(str(c) for c in categories)
        else:
            category_path = str(meta.get("main_category", ""))

        price_val = meta.get("price")
        price = float(price_val) if price_val is not None else float("nan")
        description = truncate_description(str(meta.get("description", "")))

        meta_row = item_meta.filter(pl.col("item_id") == item_id)
        first_seen = int(meta_row["first_seen_ts"][0]) if meta_row.height > 0 else 0
        n_train = int(meta_row["n_train_inter"][0]) if meta_row.height > 0 else 0

        rows.append(
            {
                "item_id": item_id,
                "raw_id": raw_id,
                "title": title,
                "brand": brand if brand else "",
                "category_path": category_path,
                "price": price,
                "description": description,
                "first_seen_ts": first_seen,
                "n_train_inter": n_train,
            }
        )
    return pl.DataFrame(rows)


def build_users_table(
    interactions: pl.DataFrame,
    user_raw_map: dict[str, str],
    user_meta: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for user_id_str, raw_id in user_raw_map.items():
        user_id = int(user_id_str)
        meta_row = user_meta.filter(pl.col("user_id") == user_id)
        rows.append(
            {
                "user_id": user_id,
                "raw_id": raw_id,
                "n_inter": int(meta_row["n_inter"][0]),
                "first_ts": int(meta_row["first_ts"][0]),
                "last_ts": int(meta_row["last_ts"][0]),
                "repeat_ratio": float(meta_row["repeat_ratio"][0]),
            }
        )
    return pl.DataFrame(rows)


def prepare_from_records(
    review_records: list[dict[str, Any]],
    meta_records: list[dict[str, Any]],
    output_dir: Path,
    split_strategy: SplitStrategy = "global_temporal",
    cold_threshold: int = 5,
    min_core: int = MIN_CORE,
) -> Path:
    """Full pipeline from raw records to parquet bundle (used by CLI and tests)."""
    logger.info("Building interactions from %d review records", len(review_records))
    interactions = build_interactions_from_records(review_records)
    interactions = filter_5core(interactions, min_core=min_core)
    interactions, user_raw_map, item_raw_map = remap_ids(interactions)
    interactions = apply_split(interactions, split_strategy)

    item_meta = compute_item_metadata(interactions)
    user_meta = compute_user_metadata(interactions)
    items = build_items_from_records(meta_records, item_raw_map, item_meta)
    users = build_users_table(interactions, user_raw_map, user_meta)
    samples = build_samples(interactions, items, split_strategy, cold_threshold=cold_threshold)

    write_parquet_bundle(output_dir, interactions, items, users, samples)
    logger.info("Wrote parquet bundle to %s", output_dir)
    return output_dir


HfRecords = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def load_amazon_category_from_hf(category: str) -> HfRecords:
    """Load Amazon Reviews 2023 category from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets package is required for HF loading. Install with: pip install datasets"
        ) from exc

    logger.info("Loading Amazon Reviews 2023 category=%s from HuggingFace", category)
    try:
        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name="raw_review",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        meta = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name="raw_meta",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Amazon Reviews 2023 from HuggingFace: {exc}") from exc

    review_records: list[dict[str, Any]] = []
    meta_records: list[dict[str, Any]] = []
    meta_asins: set[str] = set()

    for row in reviews:
        if row.get("main_category") == category or category in str(row.get("categories", "")):
            review_records.append(dict(row))

    for row in meta:
        if row.get("main_category") == category:
            meta_records.append(dict(row))
            meta_asins.add(row["parent_asin"])

    # Ensure meta exists for items in reviews
    review_asins = {r["parent_asin"] for r in review_records}
    missing = review_asins - meta_asins
    if missing:
        logger.warning("%d items missing metadata, using placeholders", len(missing))
        for asin in missing:
            meta_records.append({"parent_asin": asin, "title": asin, "main_category": category})

    if not review_records:
        raise ValueError(f"No reviews found for category '{category}'")

    return review_records, meta_records


def prepare_amazon_dataset(
    category: str,
    output_dir: Path,
    split_strategy: SplitStrategy = "global_temporal",
    cold_threshold: int = 5,
) -> Path:
    """Download and prepare Amazon Reviews 2023 for a category."""
    review_records, meta_records = load_amazon_category_from_hf(category)
    return prepare_from_records(
        review_records,
        meta_records,
        output_dir,
        split_strategy=split_strategy,
        cold_threshold=cold_threshold,
    )
