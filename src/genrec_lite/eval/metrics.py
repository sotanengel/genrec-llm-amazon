"""Evaluation metrics (DESIGN.md §8.2)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def _to_scores(scores: Any) -> FloatArray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {arr.shape}")
    return arr


def _ranking_from_scores(scores: FloatArray) -> NDArray[np.int64]:
    return np.argsort(-scores, kind="stable")


def recall_at_k(scores: Any, target: int, k: int) -> float:
    """Return 1.0 if target is in top-k, else 0.0."""
    ranking = _ranking_from_scores(_to_scores(scores))
    top_k = ranking[:k]
    return 1.0 if target in top_k else 0.0


def hit_rate_at_k(scores: Any, target: int, k: int) -> float:
    """Alias of recall@k for a single target."""
    return recall_at_k(scores, target, k)


def ndcg_at_k(scores: Any, target: int, k: int) -> float:
    """Normalized DCG@k for a single relevant item."""
    ranking = _ranking_from_scores(_to_scores(scores))
    top_k = ranking[:k]
    positions = np.where(top_k == target)[0]
    if len(positions) == 0:
        return 0.0
    rank = int(positions[0])
    dcg = 1.0 / math.log2(rank + 2)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


def mrr_at_k(scores: Any, target: int, k: int) -> float:
    """Mean reciprocal rank truncated at k for a single target."""
    ranking = _ranking_from_scores(_to_scores(scores))
    top_k = ranking[:k]
    positions = np.where(top_k == target)[0]
    if len(positions) == 0:
        return 0.0
    return 1.0 / (int(positions[0]) + 1)


def coverage_at_k(recommended: list[list[int]], n_items: int, k: int) -> float:
    """Unique recommended items / catalog size."""
    if n_items == 0:
        return 0.0
    unique: set[int] = set()
    for recs in recommended:
        unique.update(recs[:k])
    return len(unique) / n_items


def gini_at_k(recommended: list[list[int]], n_items: int, k: int) -> float:
    """Gini coefficient of item recommendation frequency."""
    counts = np.zeros(n_items, dtype=np.float64)
    for recs in recommended:
        for item in recs[:k]:
            counts[item] += 1.0
    if counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n
    )


def avg_popularity_at_k(
    recommended: list[list[int]],
    item_popularity: FloatArray,
    k: int,
) -> float:
    """Average popularity of recommended items."""
    values: list[float] = []
    for recs in recommended:
        top = recs[:k]
        if not top:
            continue
        values.append(float(np.mean(item_popularity[top])))
    if not values:
        return 0.0
    return float(np.mean(values))


def novelty_at_k(
    recommended: list[list[int]],
    item_popularity: FloatArray,
    k: int,
) -> float:
    """Average -log(popularity) of recommended items (normalized by catalog)."""
    n_items = len(item_popularity)
    if n_items == 0:
        return 0.0
    total = float(item_popularity.sum())
    if total <= 0:
        return 0.0
    probs = item_popularity / total
    values: list[float] = []
    for recs in recommended:
        top = recs[:k]
        if not top:
            continue
        novelties = [-math.log(max(float(probs[i]), 1e-12)) for i in top]
        values.append(float(np.mean(novelties)))
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_metrics_for_ranking(
    scores: Any,
    target: int,
    k: int,
) -> dict[str, float]:
    """Compute accuracy metrics for one sample."""
    return {
        f"recall@{k}": recall_at_k(scores, target, k),
        f"ndcg@{k}": ndcg_at_k(scores, target, k),
        f"mrr@{k}": mrr_at_k(scores, target, k),
        f"hit_rate@{k}": hit_rate_at_k(scores, target, k),
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts across samples."""
    if not results:
        return {}
    keys = results[0].keys()
    return {key: float(np.mean([row[key] for row in results])) for key in keys}
