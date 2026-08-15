"""Parquet schema definitions and validation (DESIGN.md §3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

import polars as pl

INTERACTIONS_SCHEMA: Final[dict[str, Any]] = {
    "user_id": pl.Int32,
    "item_id": pl.Int32,
    "ts": pl.Int64,
    "basket_id": pl.Int32,
    "rating": pl.Float32,
    "event_type": pl.Int8,
    "split": pl.Int8,
}

ITEMS_SCHEMA: Final[dict[str, Any]] = {
    "item_id": pl.Int32,
    "raw_id": pl.Utf8,
    "title": pl.Utf8,
    "brand": pl.Utf8,
    "category_path": pl.Utf8,
    "price": pl.Float32,
    "description": pl.Utf8,
    "first_seen_ts": pl.Int64,
    "n_train_inter": pl.Int32,
}

USERS_SCHEMA: Final[dict[str, Any]] = {
    "user_id": pl.Int32,
    "raw_id": pl.Utf8,
    "n_inter": pl.Int32,
    "first_ts": pl.Int64,
    "last_ts": pl.Int64,
    "repeat_ratio": pl.Float32,
}

SAMPLES_SCHEMA: Final[dict[str, Any]] = {
    "sample_id": pl.Int64,
    "user_id": pl.Int32,
    "cutoff_ts": pl.Int64,
    "target_item": pl.Int32,
    "history": pl.List(pl.Int32),
    "split": pl.Int8,
    "is_repeat": pl.Boolean,
    "target_is_cold": pl.Boolean,
}


class SchemaValidationError(ValueError):
    """Raised when a dataframe does not match the required schema."""


def _validate_schema(df: pl.DataFrame, expected: dict[str, Any], name: str) -> None:
    actual_cols = set(df.columns)
    expected_cols = set(expected.keys())
    if actual_cols != expected_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {sorted(missing)}")
        if extra:
            parts.append(f"extra columns: {sorted(extra)}")
        raise SchemaValidationError(f"{name}: {'; '.join(parts)}")

    for col, dtype in expected.items():
        if df.schema[col] != dtype:
            raise SchemaValidationError(
                f"{name}: column '{col}' has dtype {df.schema[col]}, expected {dtype}"
            )


def cast_interactions(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([pl.col(c).cast(t) for c, t in INTERACTIONS_SCHEMA.items()])


def cast_items(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([pl.col(c).cast(t) for c, t in ITEMS_SCHEMA.items()])


def cast_users(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([pl.col(c).cast(t) for c, t in USERS_SCHEMA.items()])


def cast_samples(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([pl.col(c).cast(t) for c, t in SAMPLES_SCHEMA.items()])


def validate_interactions(df: pl.DataFrame) -> None:
    _validate_schema(df, INTERACTIONS_SCHEMA, "interactions")


def validate_items(df: pl.DataFrame) -> None:
    _validate_schema(df, ITEMS_SCHEMA, "items")


def validate_users(df: pl.DataFrame) -> None:
    _validate_schema(df, USERS_SCHEMA, "users")


def validate_samples(df: pl.DataFrame) -> None:
    _validate_schema(df, SAMPLES_SCHEMA, "samples")


def assert_contiguous_ids(df: pl.DataFrame, col: str) -> None:
    ids = df[col].unique().sort().to_list()
    if not ids:
        return
    expected = list(range(len(ids)))
    if ids != expected:
        raise SchemaValidationError(
            f"Column '{col}' is not 0-indexed contiguous: got {ids[:5]}... (n={len(ids)})"
        )


def validate_bundle(
    interactions: pl.DataFrame,
    items: pl.DataFrame,
    users: pl.DataFrame,
    samples: pl.DataFrame | None = None,
) -> None:
    validate_interactions(interactions)
    validate_items(items)
    validate_users(users)
    if samples is not None:
        validate_samples(samples)

    assert_contiguous_ids(interactions, "user_id")
    assert_contiguous_ids(interactions, "item_id")
    assert_contiguous_ids(items, "item_id")
    assert_contiguous_ids(users, "user_id")

    split_values = set(interactions["split"].unique().to_list())
    if not split_values.issubset({0, 1, 2}):
        raise SchemaValidationError(f"split values must be subset of {{0,1,2}}, got {split_values}")

    ts_min = interactions["ts"].min()
    if ts_min is None or cast(int, ts_min) < 0:
        raise SchemaValidationError("timestamps must be non-negative")

    # items.first_seen_ts <= min interaction ts per item
    item_min_ts = interactions.group_by("item_id").agg(pl.col("ts").min().alias("min_ts"))
    merged = items.join(item_min_ts, on="item_id", how="left")
    bad = merged.filter(pl.col("first_seen_ts") > pl.col("min_ts"))
    if bad.height > 0:
        raise SchemaValidationError("items.first_seen_ts must be <= min interaction ts per item")

    # users.n_inter must match interaction counts
    inter_counts = interactions.group_by("user_id").len().rename({"len": "n_inter_calc"})
    user_check = users.join(inter_counts, on="user_id", how="left")
    mismatch = user_check.filter(pl.col("n_inter") != pl.col("n_inter_calc"))
    if mismatch.height > 0:
        raise SchemaValidationError("users.n_inter does not match interaction counts")


def write_parquet_bundle(
    out_dir: Path,
    interactions: pl.DataFrame,
    items: pl.DataFrame,
    users: pl.DataFrame,
    samples: pl.DataFrame,
    train_samples: pl.DataFrame | None = None,
) -> None:
    interactions = cast_interactions(interactions)
    items = cast_items(items)
    users = cast_users(users)
    samples = cast_samples(samples)
    validate_bundle(interactions, items, users, samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.write_parquet(out_dir / "interactions.parquet")
    items.write_parquet(out_dir / "items.parquet")
    users.write_parquet(out_dir / "users.parquet")
    samples.write_parquet(out_dir / "samples.parquet")
    if train_samples is not None:
        train_samples = cast_samples(train_samples)
        validate_samples(train_samples)
        train_samples.write_parquet(out_dir / "train_samples.parquet")


ParquetBundle = tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]


def read_parquet_bundle(data_dir: Path) -> ParquetBundle:
    interactions = pl.read_parquet(data_dir / "interactions.parquet")
    items = pl.read_parquet(data_dir / "items.parquet")
    users = pl.read_parquet(data_dir / "users.parquet")
    samples = pl.read_parquet(data_dir / "samples.parquet")
    validate_bundle(interactions, items, users, samples)
    return interactions, items, users, samples


def read_train_samples(data_dir: Path) -> pl.DataFrame:
    """Load train_samples.parquet; raises if missing."""
    path = data_dir / "train_samples.parquet"
    if not path.exists():
        raise FileNotFoundError(f"train_samples.parquet not found: {path}")
    samples = pl.read_parquet(path)
    validate_samples(samples)
    return samples
