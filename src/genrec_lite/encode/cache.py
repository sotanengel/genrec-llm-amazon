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
    revision: str
    verbalizer_name: str
    verbalizer_config: dict[str, Any]
    max_len: int
    dtype: str = "bfloat16"
    quantize: str | None = None
    pooling: str = "last"
    attn_implementation: str = "auto"
    deterministic: bool = False
    encoder_version: int = 1


def compute_cache_key(config: CacheKeyConfig) -> str:
    payload = json.dumps(
        {
            "model_id": config.model_id,
            "revision": config.revision,
            "verbalizer_name": config.verbalizer_name,
            "verbalizer_config": config.verbalizer_config,
            "max_len": config.max_len,
            "dtype": config.dtype,
            "quantize": config.quantize,
            "pooling": config.pooling,
            "attn_implementation": config.attn_implementation,
            "deterministic": config.deterministic,
            "encoder_version": config.encoder_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HiddenStateCache:
    """float16 memmap cache for hidden states with resumable row writes."""

    def __init__(self, cache_dir: Path, key: str, n_samples: int, hidden_dim: int) -> None:
        self.cache_dir = cache_dir
        self.key = key
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_path = self.cache_dir / f"{key}.f16.memmap"
        self.index_path = self.cache_dir / f"{key}.index.parquet"
        self.meta_path = self.cache_dir / f"{key}.meta.json"
        self.progress_path = self.cache_dir / f"{key}.progress.json"
        self._memmap: np.memmap | None = None

    @property
    def expected_bytes(self) -> int:
        return self.n_samples * self.hidden_dim * 2

    def completed_rows(self) -> set[int]:
        return self._load_completed_rows()

    def _load_completed_rows(self) -> set[int]:
        if not self.progress_path.exists():
            return set()
        data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        rows = data.get("completed_rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid progress file: {self.progress_path}")
        return {int(r) for r in rows}

    def rows_written(self) -> int:
        return len(self._load_completed_rows())

    def _open_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.memmap_path,
                dtype=np.float16,
                mode="r+" if self.memmap_path.exists() else "w+",
                shape=(self.n_samples, self.hidden_dim),
            )
        return self._memmap

    def write_rows(self, row_indices: list[int], hidden: Tensor) -> None:
        if hidden.shape != (len(row_indices), self.hidden_dim):
            raise ValueError(
                f"Expected hidden shape ({len(row_indices)}, {self.hidden_dim}), "
                f"got {tuple(hidden.shape)}"
            )
        memmap = self._open_memmap()
        arr = hidden.detach().cpu().to(torch.float16).numpy()
        for offset, row_idx in enumerate(row_indices):
            if row_idx < 0 or row_idx >= self.n_samples:
                raise IndexError(f"row_idx {row_idx} out of range for n_samples={self.n_samples}")
            memmap[row_idx] = arr[offset]
        memmap.flush()
        completed = self._load_completed_rows()
        completed.update(row_indices)
        self.progress_path.write_text(
            json.dumps({"completed_rows": sorted(completed)}),
            encoding="utf-8",
        )

    def finalize(self, sample_ids: list[int], meta: dict[str, Any] | None = None) -> None:
        if len(sample_ids) != self.n_samples:
            raise ValueError(f"Expected {self.n_samples} sample_ids, got {len(sample_ids)}")
        if self.rows_written() != self.n_samples:
            raise ValueError(
                f"Cannot finalize: wrote {self.rows_written()} rows, expected {self.n_samples}"
            )
        pl.DataFrame(
            {"sample_id": sample_ids, "row_idx": list(range(len(sample_ids)))}
        ).write_parquet(self.index_path)
        sidecar = meta or {}
        self.meta_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        if self.progress_path.exists():
            self.progress_path.unlink()

    def save(
        self, sample_ids: list[int], hidden: Tensor, meta: dict[str, Any] | None = None
    ) -> None:
        row_indices = list(range(self.n_samples))
        self.write_rows(row_indices, hidden)
        self.finalize(sample_ids, meta)

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
        return self.memmap_path.exists() and self.index_path.exists() and self.meta_path.exists()
