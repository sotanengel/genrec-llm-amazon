"""Popularity baseline (G-TopFreq)."""

from __future__ import annotations

import numpy as np
import polars as pl


class PopRecommender:
    """Global top-frequency baseline."""

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        train = interactions.filter(pl.col("split") == 0)
        n_items = items.height
        counts = np.zeros(n_items, dtype=np.float64)
        if train.height > 0:
            grouped = train.group_by("item_id").len()
            for item_id, count in grouped.iter_rows():
                counts[int(item_id)] = float(count)
        self._scores = counts

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("Model not fitted")
        return np.tile(self._scores, (samples.height, 1))

    def name(self) -> str:
        return "pop"
