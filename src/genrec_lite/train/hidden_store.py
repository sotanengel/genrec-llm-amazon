"""Memmap-backed hidden state lookup for head training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import Tensor

from genrec_lite.encode.cache import HiddenStateCache


class HiddenStateStore:
    """Read-only view over a finalized hidden-state cache."""

    def __init__(self, cache: HiddenStateCache) -> None:
        if not cache.exists():
            msg = f"Hidden state cache is not finalized: {cache.memmap_path}. Run encode first."
            raise FileNotFoundError(msg)
        self._cache = cache
        index_df = pl.read_parquet(cache.index_path)
        self._sample_id_to_row = {
            int(row["sample_id"]): int(row["row_idx"]) for row in index_df.iter_rows(named=True)
        }
        self._memmap = np.memmap(
            cache.memmap_path,
            dtype=np.float16,
            mode="r",
            shape=(cache.n_samples, cache.hidden_dim),
        )

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: Path,
        key: str,
        hidden_dim: int,
        scope: str = "eval",
    ) -> HiddenStateStore:
        from genrec_lite.encode.cache import CacheScope

        cache_scope: CacheScope = scope  # type: ignore[assignment]
        cache = HiddenStateCache.open_existing(cache_dir, key, hidden_dim, scope=cache_scope)
        return cls(cache)

    @property
    def hidden_dim(self) -> int:
        return self._cache.hidden_dim

    def get_vectors(self, sample_ids: list[int]) -> Tensor:
        """Fetch hidden states for the given sample ids."""
        rows = [self._sample_id_to_row[sid] for sid in sample_ids]
        vectors = np.asarray(self._memmap[rows], dtype=np.float32)
        return torch.from_numpy(vectors)

    def has_sample(self, sample_id: int) -> bool:
        return sample_id in self._sample_id_to_row
