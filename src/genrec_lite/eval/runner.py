"""Evaluation runner (DESIGN.md §8)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from genrec_lite.eval.metrics import (
    aggregate_metrics,
    avg_popularity_at_k,
    coverage_at_k,
    gini_at_k,
    novelty_at_k,
)
from genrec_lite.eval.slices import all_slice_names, assign_slice, filter_slice


def _scores_to_rankings(scores: np.ndarray, max_k: int) -> list[list[int]]:
    return [list(np.argsort(-row, kind="stable")[:max_k]) for row in scores]


def _accuracy_metrics(ranking: list[int], target: int, ks: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        rank = ranking.index(target)
    except ValueError:
        rank = -1
    for k in ks:
        hit = rank >= 0 and rank < k
        metrics[f"recall@{k}"] = float(hit)
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 2) if hit else 0.0
        metrics[f"mrr@{k}"] = 1.0 / (rank + 1) if hit else 0.0
        metrics[f"hit_rate@{k}"] = float(hit)
    return metrics


def evaluate(
    score_fn: Callable[[pl.DataFrame], Any],
    samples: pl.DataFrame,
    items: pl.DataFrame,
    interactions: pl.DataFrame,
    ks: tuple[int, ...] = (10, 20),
    slices: tuple[str, ...] | None = None,
    cold_threshold: int = 5,
    method: str = "unknown",
    eval_batch_size: int = 32,
) -> pd.DataFrame:
    """Evaluate a scoring function with full-catalog ranking."""
    if eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")
    if slices is None:
        slices = all_slice_names(cold_threshold=cold_threshold)
    n_items = items.height
    max_k = max(ks, default=0)
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

        rankings: list[list[int]] = []
        metric_rows: list[dict[str, float]] = []
        for offset in range(0, slice_df.height, eval_batch_size):
            batch = slice_df.slice(offset, eval_batch_size)
            scores = np.asarray(score_fn(batch), dtype=np.float64)
            expected_shape = (batch.height, n_items)
            if scores.ndim != 2 or scores.shape != expected_shape:
                raise ValueError(
                    f"Full-catalog ranking required: expected shape {expected_shape}, "
                    f"got {scores.shape}"
                )

            batch_rankings = _scores_to_rankings(scores, max_k)
            rankings.extend(batch_rankings)
            for i in range(batch.height):
                target = int(batch["target_item"][i])
                metric_rows.append(_accuracy_metrics(batch_rankings[i], target, ks))

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
