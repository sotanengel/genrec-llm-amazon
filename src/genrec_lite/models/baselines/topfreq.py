"""Personal and global-personal top-frequency baselines."""

from __future__ import annotations

import numpy as np
import polars as pl


class PTopFreqRecommender:
    """Personal top-frequency baseline."""

    def __init__(self) -> None:
        self._user_counts: dict[int, np.ndarray] = {}
        self._n_items: int = 0

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        self._n_items = items.height
        train = interactions.filter(pl.col("split") == 0)
        self._user_counts = {}
        for user_id in train["user_id"].unique().to_list():
            user_inter = train.filter(pl.col("user_id") == user_id)
            counts = np.zeros(self._n_items, dtype=np.float64)
            grouped = user_inter.group_by("item_id").len()
            for item_id, count in grouped.iter_rows():
                counts[int(item_id)] = float(count)
            self._user_counts[int(user_id)] = counts

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        scores = np.zeros((samples.height, self._n_items), dtype=np.float64)
        for i, row in enumerate(samples.iter_rows(named=True)):
            user_id = int(row["user_id"])
            scores[i] = self._user_counts.get(user_id, np.zeros(self._n_items))
        return scores

    def name(self) -> str:
        return "p_topfreq"


class GPTopFreqRecommender:
    """Personal frequency with global fallback."""

    def __init__(self) -> None:
        self._global_scores: np.ndarray | None = None
        self._user_counts: dict[int, np.ndarray] = {}
        self._n_items: int = 0

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        self._n_items = items.height
        train = interactions.filter(pl.col("split") == 0)
        global_counts = np.zeros(self._n_items, dtype=np.float64)
        grouped = train.group_by("item_id").len()
        for item_id, count in grouped.iter_rows():
            global_counts[int(item_id)] = float(count)
        self._global_scores = global_counts

        self._user_counts = {}
        for user_id in train["user_id"].unique().to_list():
            user_inter = train.filter(pl.col("user_id") == user_id)
            counts = np.zeros(self._n_items, dtype=np.float64)
            user_grouped = user_inter.group_by("item_id").len()
            for item_id, count in user_grouped.iter_rows():
                counts[int(item_id)] = float(count)
            self._user_counts[int(user_id)] = counts

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        if self._global_scores is None:
            raise RuntimeError("Model not fitted")
        scores = np.tile(self._global_scores, (samples.height, 1))
        for i, row in enumerate(samples.iter_rows(named=True)):
            user_id = int(row["user_id"])
            history = row["history"]
            user_counts = self._user_counts.get(user_id)
            if user_counts is not None and len(history) >= 2:
                scores[i] = user_counts
        return scores

    def name(self) -> str:
        return "gp_topfreq"
