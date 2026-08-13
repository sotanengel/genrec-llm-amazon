"""Train/valid/test splitting and sample generation (DESIGN.md §3.3)."""

from __future__ import annotations

from typing import Literal

import polars as pl

SplitStrategy = Literal["leave_one_out", "global_temporal"]

SPLIT_TRAIN = 0
SPLIT_VALID = 1
SPLIT_TEST = 2


def apply_leave_one_out(interactions: pl.DataFrame) -> pl.DataFrame:
    """Per-user: last interaction -> test, second-last -> valid, rest -> train."""
    ranked = interactions.with_columns(
        pl.col("ts").rank(method="ordinal", descending=True).over("user_id").alias("_rank_desc")
    )
    return ranked.with_columns(
        pl.when(pl.col("_rank_desc") == 1)
        .then(pl.lit(SPLIT_TEST))
        .when(pl.col("_rank_desc") == 2)
        .then(pl.lit(SPLIT_VALID))
        .otherwise(pl.lit(SPLIT_TRAIN))
        .cast(pl.Int8)
        .alias("split")
    ).drop("_rank_desc")


def apply_global_temporal(
    interactions: pl.DataFrame,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> pl.DataFrame:
    """Split by global timestamp: earliest -> train, middle -> valid, latest -> test."""
    sorted_ts = interactions.select("ts").unique().sort("ts")
    n_ts = sorted_ts.height
    if n_ts < 3:
        raise ValueError("Need at least 3 unique timestamps for global_temporal split")

    test_start_idx = max(1, int(n_ts * (1.0 - test_ratio)))
    valid_start_idx = max(1, int(n_ts * (1.0 - test_ratio - valid_ratio)))

    test_cutoff = sorted_ts["ts"][test_start_idx]
    valid_cutoff = sorted_ts["ts"][valid_start_idx]

    return interactions.with_columns(
        pl.when(pl.col("ts") >= test_cutoff)
        .then(pl.lit(SPLIT_TEST))
        .when(pl.col("ts") >= valid_cutoff)
        .then(pl.lit(SPLIT_VALID))
        .otherwise(pl.lit(SPLIT_TRAIN))
        .cast(pl.Int8)
        .alias("split")
    )


def apply_split(interactions: pl.DataFrame, strategy: SplitStrategy) -> pl.DataFrame:
    if strategy == "leave_one_out":
        return apply_leave_one_out(interactions)
    if strategy == "global_temporal":
        return apply_global_temporal(interactions)
    raise ValueError(f"Unknown split strategy: {strategy}")


def compute_item_metadata(interactions: pl.DataFrame) -> pl.DataFrame:
    """Compute first_seen_ts and n_train_inter from train-split interactions."""
    train = interactions.filter(pl.col("split") == SPLIT_TRAIN)
    return train.group_by("item_id").agg(
        pl.col("ts").min().alias("first_seen_ts"),
        pl.len().alias("n_train_inter"),
    )


def compute_user_metadata(interactions: pl.DataFrame) -> pl.DataFrame:
    """Compute user-level stats including repeat_ratio."""
    user_stats = interactions.group_by("user_id").agg(
        pl.len().alias("n_inter"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
        pl.col("item_id").n_unique().alias("n_unique_items"),
    )
    return user_stats.with_columns(
        pl.when(pl.col("n_inter") <= 1)
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(
            ((pl.col("n_inter") - pl.col("n_unique_items")).cast(pl.Float32))
            / ((pl.col("n_inter") - 1).cast(pl.Float32))
        )
        .alias("repeat_ratio")
    ).drop("n_unique_items")


def build_samples(
    interactions: pl.DataFrame,
    items: pl.DataFrame,
    strategy: SplitStrategy,
    cold_threshold: int = 5,
) -> pl.DataFrame:
    """Build evaluation/training samples from split interactions."""
    eval_interactions = interactions.filter(pl.col("split").is_in([SPLIT_VALID, SPLIT_TEST]))
    item_cold = items.select("item_id", "n_train_inter")

    rows: list[dict[str, object]] = []
    sample_id = 0

    for user_id in eval_interactions["user_id"].unique().sort().to_list():
        user_all = interactions.filter(pl.col("user_id") == user_id).sort("ts")
        user_eval = eval_interactions.filter(pl.col("user_id") == user_id).sort("ts")

        for eval_row in user_eval.iter_rows(named=True):
            cutoff_ts = eval_row["ts"]
            target_item = eval_row["item_id"]
            split_val = eval_row["split"]

            history_df = user_all.filter(pl.col("ts") < cutoff_ts)
            history = history_df["item_id"].to_list()
            is_repeat = target_item in history

            cold_info = item_cold.filter(pl.col("item_id") == target_item)
            n_train = cold_info["n_train_inter"][0] if cold_info.height > 0 else 0
            target_is_cold = n_train < cold_threshold

            rows.append(
                {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "cutoff_ts": cutoff_ts,
                    "target_item": target_item,
                    "history": history,
                    "split": split_val,
                    "is_repeat": is_repeat,
                    "target_is_cold": target_is_cold,
                }
            )
            sample_id += 1

    return pl.DataFrame(
        rows,
        schema={
            "sample_id": pl.Int64,
            "user_id": pl.Int32,
            "cutoff_ts": pl.Int64,
            "target_item": pl.Int32,
            "history": pl.List(pl.Int32),
            "split": pl.Int8,
            "is_repeat": pl.Boolean,
            "target_is_cold": pl.Boolean,
        },
    )
