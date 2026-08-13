"""Shared pytest fixtures (DESIGN.md §14.1)."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest
from genrec_lite.data.loaders.amazon import prepare_from_records
from genrec_lite.data.schema import ParquetBundle, read_parquet_bundle
from genrec_lite.data.split import apply_split

TINY_MODEL_ID = "sshleifer/tiny-gpt2"


def _make_review_records(n_users: int = 10, n_items: int = 20, n_inter: int = 80) -> list[dict]:
    """Generate synthetic Amazon-like review records for testing."""
    random.seed(42)
    records: list[dict] = []
    base_ts = 1_600_000_000_000  # ms timestamps
    for i in range(n_inter):
        user_idx = random.randint(0, n_users - 1)
        item_idx = random.randint(0, n_items - 1)
        records.append(
            {
                "user_id": f"user_{user_idx}",
                "parent_asin": f"item_{item_idx}",
                "timestamp": base_ts + i * 86_400_000,
                "rating": float(random.randint(1, 5)),
                "main_category": "Video_Games",
            }
        )
    return records


def _make_meta_records(n_items: int = 20) -> list[dict]:
    records: list[dict] = []
    for i in range(n_items):
        records.append(
            {
                "parent_asin": f"item_{i}",
                "title": f"Game Title {i}",
                "store": f"Brand{i}",
                "main_category": "Video_Games",
                "categories": [["Electronics", "Video Games", f"Subcat{i}"]],
                "price": 29.99 + i,
                "description": f"Description for item {i} " * 10,
            }
        )
    return records


@pytest.fixture
def deterministic_seeds() -> None:
    random.seed(42)


@pytest.fixture
def tiny_model_id() -> str:
    return TINY_MODEL_ID


@pytest.fixture
def mini_review_records() -> list[dict]:
    return _make_review_records()


@pytest.fixture
def mini_meta_records() -> list[dict]:
    return _make_meta_records()


@pytest.fixture
def mini_dataset(
    tmp_path: Path,
    mini_review_records: list[dict],
    mini_meta_records: list[dict],
) -> Path:
    """Mini parquet bundle: ~10 users, ~20 items, 80 interactions."""
    out_dir = tmp_path / "mini"
    prepare_from_records(
        mini_review_records,
        mini_meta_records,
        out_dir,
        split_strategy="global_temporal",
        cold_threshold=5,
        min_core=3,  # Lower for mini fixture size
    )
    return out_dir


@pytest.fixture
def mini_dataset_loo(
    tmp_path: Path,
    mini_review_records: list[dict],
    mini_meta_records: list[dict],
) -> Path:
    out_dir = tmp_path / "mini_loo"
    prepare_from_records(
        mini_review_records,
        mini_meta_records,
        out_dir,
        split_strategy="leave_one_out",
        cold_threshold=5,
        min_core=3,
    )
    return out_dir


@pytest.fixture
def mini_interactions(
    mini_review_records: list[dict],
    mini_meta_records: list[dict],
) -> pl.DataFrame:
    from genrec_lite.data.loaders.amazon import (
        build_interactions_from_records,
        filter_5core,
        remap_ids,
    )

    interactions = build_interactions_from_records(mini_review_records)
    interactions = filter_5core(interactions, min_core=3)
    interactions, _, _ = remap_ids(interactions)
    return interactions


@pytest.fixture
def mini_interactions_global(mini_interactions: pl.DataFrame) -> pl.DataFrame:
    return apply_split(mini_interactions, "global_temporal")


@pytest.fixture
def mini_interactions_loo(mini_interactions: pl.DataFrame) -> pl.DataFrame:
    return apply_split(mini_interactions, "leave_one_out")


@pytest.fixture
def mini_bundle(mini_dataset: Path) -> ParquetBundle:
    return read_parquet_bundle(mini_dataset)
