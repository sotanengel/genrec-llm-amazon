"""Hidden state cache with memmap (DESIGN.md §10.8)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheKeyConfig:
    model_id: str
    verbalizer_name: str
    verbalizer_config: dict[str, Any]
    max_len: int


def compute_cache_key(config: CacheKeyConfig) -> str:
    payload = json.dumps(
        {
            "model_id": config.model_id,
            "verbalizer_name": config.verbalizer_name,
            "verbalizer_config": config.verbalizer_config,
            "max_len": config.max_len,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HiddenStateCache:
    """float16 memmap cache for hidden states."""

    def __init__(self, cache_dir: Path, key: str, n_samples: int, hidden_dim: int) -> None:
        self.cache_dir = cache_dir
        self.key = key
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_path = self.cache_dir / f"{key}.f16.memmap"
        self.index_path = self.cache_dir / f"{key}.index.parquet"
        self._memmap: np.memmap | None = None

    @property
    def expected_bytes(self) -> int:
        return self.n_samples * self.hidden_dim * 2

    def _open_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.memmap_path,
                dtype=np.float16,
                mode="r+" if self.memmap_path.exists() else "w+",
                shape=(self.n_samples, self.hidden_dim),
            )
        return self._memmap

    def save(self, sample_ids: list[int], hidden: Tensor) -> None:
        arr = hidden.detach().cpu().numpy().astype(np.float16)
        if arr.shape != (self.n_samples, self.hidden_dim):
            raise ValueError(
                f"Expected hidden shape ({self.n_samples}, {self.hidden_dim}), got {arr.shape}"
            )
        memmap = self._open_memmap()
        memmap[:] = arr
        memmap.flush()
        pl.DataFrame(
            {"sample_id": sample_ids, "row_idx": list(range(len(sample_ids)))}
        ).write_parquet(self.index_path)

    def load(self) -> Tensor:
        if not self.memmap_path.exists():
            raise FileNotFoundError(f"Cache memmap not found: {self.memmap_path}")
        actual_bytes = self.memmap_path.stat().st_size
        if actual_bytes != self.expected_bytes:
            raise ValueError(
                f"Cache size mismatch: expected {self.expected_bytes}, got {actual_bytes}"
            )
        memmap = self._open_memmap()
        return torch.from_numpy(np.array(memmap, copy=True))

    def exists(self) -> bool:
        return self.memmap_path.exists() and self.index_path.exists()
