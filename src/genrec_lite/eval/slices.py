"""Evaluation slice definitions (DESIGN.md §8.3)."""

from __future__ import annotations

import polars as pl

DEFAULT_SLICES: tuple[str, ...] = (
    "all",
    "repeat",
    "explore",
    "cold",
    "warm",
    "short_hist",
    "long_hist",
)


def _popularity_deciles(
    samples: pl.DataFrame,
    interactions: pl.DataFrame,
) -> pl.Series:
    train = interactions.filter(pl.col("split") == 0)
    item_counts = train.group_by("item_id").len().rename({"len": "pop"})
    merged = samples.join(item_counts, left_on="target_item", right_on="item_id", how="left")
    pops = merged["pop"].fill_null(0).to_numpy()
    if len(pops) == 0:
        return pl.Series("pop_decile", [], dtype=pl.Utf8)
    unique_pops = sorted(set(pops.tolist()))
    if len(unique_pops) <= 1:
        return pl.Series("pop_decile", ["pop_decile_0"] * len(pops))
    deciles: list[str] = []
    for pop in pops:
        rank = unique_pops.index(pop)
        bucket = min(9, int(rank * 10 / len(unique_pops)))
        deciles.append(f"pop_decile_{bucket}")
    return pl.Series("pop_decile", deciles)


def assign_slice(
    samples: pl.DataFrame,
    items: pl.DataFrame,
    interactions: pl.DataFrame,
    cold_threshold: int = 5,
) -> pl.DataFrame:
    """Add a `slice_name` column for each sample (primary slice)."""
    item_cold = items.select(["item_id", "n_train_inter"]).rename(
        {"n_train_inter": "target_n_train_inter"}
    )
    enriched = samples.join(item_cold, left_on="target_item", right_on="item_id", how="left")
    enriched = enriched.with_columns(
        pl.col("history").list.len().alias("hist_len"),
        pl.col("target_n_train_inter").fill_null(0),
    )
    pop_decile = _popularity_deciles(samples, interactions)
    enriched = enriched.with_columns(pop_decile)
    return enriched


def filter_slice(df: pl.DataFrame, slice_name: str, cold_threshold: int = 5) -> pl.DataFrame:
    """Return rows belonging to the given slice."""
    if slice_name == "all":
        return df
    if slice_name == "repeat":
        return df.filter(pl.col("is_repeat"))
    if slice_name == "explore":
        return df.filter(~pl.col("is_repeat"))
    if slice_name == "cold":
        return df.filter(pl.col("target_n_train_inter") < cold_threshold)
    if slice_name == "warm":
        return df.filter(pl.col("target_n_train_inter") >= 20)
    if slice_name == "short_hist":
        return df.filter(pl.col("hist_len") < 5)
    if slice_name == "long_hist":
        return df.filter(pl.col("hist_len") >= 20)
    if slice_name.startswith("pop_decile_"):
        return df.filter(pl.col("pop_decile") == slice_name)
    raise ValueError(f"Unknown slice: {slice_name}")


def all_slice_names(cold_threshold: int = 5) -> tuple[str, ...]:
    """Return default slices plus popularity deciles."""
    deciles = tuple(f"pop_decile_{i}" for i in range(10))
    return (*DEFAULT_SLICES, *deciles)
