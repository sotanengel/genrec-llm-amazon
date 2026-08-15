"""Train/valid/test splitting and sample generation (DESIGN.md §3.3)."""

from __future__ import annotations

from typing import Any, Literal

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
    """Compute per-item metadata.

    `first_seen_ts` is the earliest timestamp across ALL splits (train / valid /
    test), which matches the invariant enforced by the schema validator
    (`items.first_seen_ts <= min interaction ts per item`). Computing it over
    train alone would violate the invariant for items whose earliest
    interaction happens to have been assigned to a user's leave-one-out
    valid/test event.

    `n_train_inter` counts only train-split interactions (items with zero train
    interactions are still included with a 0 count so every item that appears
    anywhere in `interactions` gets a row).
    """
    first_seen = interactions.group_by("item_id").agg(
        pl.col("ts").min().alias("first_seen_ts"),
    )
    train_counts = (
        interactions.filter(pl.col("split") == SPLIT_TRAIN)
        .group_by("item_id")
        .agg(pl.len().alias("n_train_inter"))
    )
    return first_seen.join(train_counts, on="item_id", how="left").with_columns(
        pl.col("n_train_inter").fill_null(0).cast(pl.Int64)
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


_SAMPLES_SCHEMA: dict[str, Any] = {
    "sample_id": pl.Int64,
    "user_id": pl.Int32,
    "cutoff_ts": pl.Int64,
    "target_item": pl.Int32,
    "history": pl.List(pl.Int32),
    "split": pl.Int8,
    "is_repeat": pl.Boolean,
    "target_is_cold": pl.Boolean,
}


def build_train_samples(
    interactions: pl.DataFrame,
    items: pl.DataFrame,
    cold_threshold: int = 5,
) -> pl.DataFrame:
    """Build training samples from train-split interactions only.

    Each train interaction becomes one sample. History contains only prior
    train interactions for the same user (ts strictly before cutoff). The user's
    first train event is excluded because it has empty history.

    ``sample_id`` is assigned independently in ``user_id``, ``ts`` sort order
    starting at 0. Eval samples in ``samples.parquet`` keep their own ID space.
    """
    train = interactions.filter(pl.col("split") == SPLIT_TRAIN).sort(["user_id", "ts"])
    if train.height == 0:
        return pl.DataFrame(schema=_SAMPLES_SCHEMA)

    item_cold = items.select("item_id", "n_train_inter")
    hist_src = train.select(
        pl.col("user_id"),
        pl.col("ts").alias("hist_ts"),
        pl.col("item_id").alias("hist_item"),
    )
    targets = train.select(
        pl.col("user_id"),
        pl.col("ts").alias("cutoff_ts"),
        pl.col("item_id").alias("target_item"),
    )

    with_history = (
        targets.join(hist_src, on="user_id", how="inner")
        .filter(pl.col("hist_ts") < pl.col("cutoff_ts"))
        .group_by(["user_id", "cutoff_ts", "target_item"])
        .agg(pl.col("hist_item").sort_by("hist_ts").alias("history"))
    )

    samples = (
        with_history.sort(["user_id", "cutoff_ts"])
        .with_row_index("sample_id")
        .with_columns(
            pl.lit(SPLIT_TRAIN).cast(pl.Int8).alias("split"),
            pl.col("history").list.contains(pl.col("target_item")).alias("is_repeat"),
        )
        .join(item_cold, left_on="target_item", right_on="item_id", how="left")
        .with_columns(
            pl.col("n_train_inter").fill_null(0).lt(cold_threshold).alias("target_is_cold"),
        )
        .select(
            pl.col("sample_id").cast(pl.Int64),
            "user_id",
            "cutoff_ts",
            "target_item",
            "history",
            "split",
            "is_repeat",
            "target_is_cold",
        )
    )
    return samples
