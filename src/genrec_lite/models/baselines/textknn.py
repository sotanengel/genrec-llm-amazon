"""TextKNN content-based baseline (DESIGN.md §6)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl


def _default_item_text(item: dict[str, object]) -> str:
    title = str(item.get("title", ""))
    brand = str(item.get("brand", ""))
    category = str(item.get("category_path", ""))
    return f"{title} {brand} {category}".strip()


class TextKNNRecommender:
    """Average history text embeddings vs all item embeddings."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._item_embeddings: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        self._n_items = items.height
        texts = [_default_item_text(row) for row in items.iter_rows(named=True)]
        if self._embed_fn is None:
            rng = np.random.default_rng(42)
            self._item_embeddings = rng.standard_normal((self._n_items, 16))
        else:
            self._item_embeddings = np.asarray(self._embed_fn(texts), dtype=np.float64)
        norms = np.linalg.norm(self._item_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._item_embeddings = self._item_embeddings / norms

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        if self._item_embeddings is None:
            raise RuntimeError("Model not fitted")
        scores = np.zeros((samples.height, self._n_items), dtype=np.float64)
        for i, row in enumerate(samples.iter_rows(named=True)):
            history = [int(h) for h in row["history"]]
            if not history:
                continue
            hist_emb = self._item_embeddings[history].mean(axis=0)
            hist_norm = np.linalg.norm(hist_emb)
            if hist_norm > 0:
                hist_emb = hist_emb / hist_norm
            scores[i] = self._item_embeddings @ hist_emb
        return scores

    def name(self) -> str:
        return "textknn"
