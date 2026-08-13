"""Evaluation runner (DESIGN.md §8)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from genrec_lite.eval.metrics import (
    aggregate_metrics,
    avg_popularity_at_k,
    compute_metrics_for_ranking,
    coverage_at_k,
    gini_at_k,
    novelty_at_k,
)
from genrec_lite.eval.slices import all_slice_names, assign_slice, filter_slice


def _scores_to_rankings(scores: np.ndarray) -> list[list[int]]:
    return [list(np.argsort(-row, kind="stable")) for row in scores]


def evaluate(
    score_fn: Callable[[pl.DataFrame], Any],
    samples: pl.DataFrame,
    items: pl.DataFrame,
    interactions: pl.DataFrame,
    ks: tuple[int, ...] = (10, 20),
    slices: tuple[str, ...] | None = None,
    cold_threshold: int = 5,
    method: str = "unknown",
) -> pd.DataFrame:
    """Evaluate a scoring function with full-catalog ranking."""
    if slices is None:
        slices = all_slice_names(cold_threshold=cold_threshold)
    n_items = items.height
    enriched = assign_slice(samples, items, interactions, cold_threshold=cold_threshold)

    train = interactions.filter(pl.col("split") == 0)
    item_pop = np.zeros(n_items, dtype=np.float64)
    if train.height > 0:
        counts = train.group_by("item_id").len()
        for row in counts.iter_rows():
            item_id, count = row[0], row[1]
            item_pop[int(item_id)] = float(count)

    rows: list[dict[str, Any]] = []
    for slice_name in slices:
        slice_df = filter_slice(enriched, slice_name, cold_threshold=cold_threshold)
        if slice_df.height == 0:
            continue

        scores = np.asarray(score_fn(slice_df), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != n_items:
            raise ValueError(
                f"Full-catalog ranking required: expected shape (B, {n_items}), got {scores.shape}"
            )

        rankings = _scores_to_rankings(scores)
        metric_rows: list[dict[str, float]] = []
        for i in range(slice_df.height):
            target = int(slice_df["target_item"][i])
            sample_metrics: dict[str, float] = {}
            for k in ks:
                sample_metrics.update(compute_metrics_for_ranking(scores[i], target, k))
            metric_rows.append(sample_metrics)

        agg = aggregate_metrics(metric_rows)
        for k in ks:
            agg[f"coverage@{k}"] = coverage_at_k(rankings, n_items, k)
            agg[f"gini@{k}"] = gini_at_k(rankings, n_items, k)
            agg[f"avg_popularity@{k}"] = avg_popularity_at_k(rankings, item_pop, k)
            agg[f"novelty@{k}"] = novelty_at_k(rankings, item_pop, k)

        row_dict: dict[str, Any] = {
            "method": method,
            "slice": slice_name,
            "n_samples": slice_df.height,
        }
        row_dict.update(agg)
        rows.append(row_dict)

    return pd.DataFrame(rows)
