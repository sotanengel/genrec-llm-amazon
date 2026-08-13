"""ItemKNN co-occurrence baseline."""

from __future__ import annotations

import numpy as np
import polars as pl


class ItemKNNRecommender:
    """Co-occurrence based item-item collaborative filtering."""

    def __init__(self, k_neighbors: int = 10) -> None:
        self._k_neighbors = k_neighbors
        self._similarity: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        self._n_items = items.height
        train = interactions.filter(pl.col("split") == 0)
        cooc = np.zeros((self._n_items, self._n_items), dtype=np.float64)

        for user_id in train["user_id"].unique().to_list():
            user_items = train.filter(pl.col("user_id") == user_id)["item_id"].to_list()
            unique_items = list(set(int(i) for i in user_items))
            for i in unique_items:
                for j in unique_items:
                    if i != j:
                        cooc[i, j] += 1.0

        norms_i = np.linalg.norm(cooc, axis=1)
        norms_j = np.linalg.norm(cooc, axis=0)
        denom = np.outer(norms_i, norms_j)
        denom = np.where(denom == 0, 1.0, denom)
        self._similarity = cooc / np.sqrt(denom)

    def _item_similarity(self, i: int, j: int) -> float:
        if self._similarity is None:
            return 0.0
        return float(self._similarity[i, j])

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        if self._similarity is None:
            raise RuntimeError("Model not fitted")
        scores = np.zeros((samples.height, self._n_items), dtype=np.float64)
        for row_idx, row in enumerate(samples.iter_rows(named=True)):
            history = [int(i) for i in row["history"]]
            if not history:
                continue
            for item_id in range(self._n_items):
                scores[row_idx, item_id] = sum(
                    self._similarity[item_id, h] for h in history if h != item_id
                )
        return scores

    def name(self) -> str:
        return "itemknn"
