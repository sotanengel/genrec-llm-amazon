"""Hidden state cache with memmap (DESIGN.md §10.8)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from torch import Tensor

from genrec_lite.encode.prefill import ENCODER_VERSION

# There is no import cycle here: prefill.py imports from genrec_lite.config
# and genrec_lite.encode.backend only, neither of which import this module.
# If prefill.py ever grows a dependency on cache.py, ENCODER_VERSION must be
# relocated to a neutral module rather than papering over the cycle (D3).


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
    # Defaults to the *current* encoder generation so a caller that forgets
    # to pass this explicitly cannot silently compute a cache key for a
    # stale/different encoder (D3). Callers that care about a specific
    # historical generation must still pass it explicitly.
    encoder_version: int = ENCODER_VERSION


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


CacheScope = Literal["eval", "train"]


def _scope_suffix(scope: CacheScope) -> str:
    """Filename suffix for train scope; eval keeps legacy paths for compatibility."""
    return "" if scope == "eval" else ".train"


class MemmapSizeMismatchError(ValueError):
    """Raised when an on-disk memmap size does not match the requested row count."""


class HiddenStateCache:
    """float16 memmap cache for hidden states with resumable row writes.

    Progress tracking (DESIGN.md §9 M2, issue audit D1):

    The naive approach — re-reading and re-serializing the full
    completed-rows set after every batch — is O(N) per batch, i.e. O(N^2/B)
    over a run, which becomes multi-gigabyte redundant I/O and quadratic CPU
    on a 100k-row run with small batches (especially over a 9p-mounted
    Windows filesystem).

    Instead, `write_rows` appends a single JSON line (the row indices just
    written) to an append-only log (`*.progress.jsonl`), flushed and fsynced
    so a killed process does not lose acknowledged rows. This makes
    `write_rows` O(batch_size) amortized instead of O(N). Reading the full
    completed-rows set (`completed_rows()` / `rows_written()`) is still
    O(rows-written-so-far), but that only happens once per resume/finalize,
    not once per batch.

    A fixed-size bitmap file (mmap'd, one byte per row) would also satisfy
    the O(1)-per-row requirement and would make `completed_rows()` O(1) too;
    an append-only log was chosen instead because it needs no pre-sizing,
    is trivially human-inspectable, and its crash-safety story (append +
    flush + fsync) is simpler to reason about than partial bitmap writes.

    Backward compatibility: a pre-upgrade `*.progress.json` (single JSON
    object `{"completed_rows": [...]}`) is still read on resume rather than
    silently restarting a partially finished multi-hour job. Both formats
    are merged if both happen to be present (e.g. a run upgraded mid-flight).
    """

    def __init__(
        self,
        cache_dir: Path,
        key: str,
        n_samples: int,
        hidden_dim: int,
        scope: CacheScope = "eval",
    ) -> None:
        self.cache_dir = cache_dir
        self.key = key
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.scope = scope
        suffix = _scope_suffix(scope)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_path = self.cache_dir / f"{key}{suffix}.f16.memmap"
        self.index_path = self.cache_dir / f"{key}{suffix}.index.parquet"
        self.meta_path = self.cache_dir / f"{key}{suffix}.meta.json"
        # New append-only progress log (D1). One JSON array of row indices
        # per line, appended (never rewritten) by write_rows.
        self.progress_path = self.cache_dir / f"{key}{suffix}.progress.jsonl"
        # Old whole-file progress format, kept only for backward-compatible
        # reads on resume (D1).
        self._legacy_progress_path = self.cache_dir / f"{key}{suffix}.progress.json"
        self._memmap: np.memmap | None = None

    @staticmethod
    def infer_n_samples(memmap_path: Path, hidden_dim: int) -> int:
        """Derive row count from an on-disk float16 memmap file."""
        if not memmap_path.exists():
            raise FileNotFoundError(f"Cache memmap not found: {memmap_path}")
        nbytes = memmap_path.stat().st_size
        row_bytes = hidden_dim * 2
        if nbytes % row_bytes != 0:
            raise ValueError(
                f"Memmap size {nbytes} is not a multiple of hidden_dim*2 ({row_bytes})"
            )
        return nbytes // row_bytes

    @classmethod
    def open_existing(
        cls,
        cache_dir: Path,
        key: str,
        hidden_dim: int,
        scope: CacheScope = "eval",
    ) -> HiddenStateCache:
        """Open a finalized cache, inferring ``n_samples`` from the memmap file."""
        suffix = _scope_suffix(scope)
        memmap_path = cache_dir / f"{key}{suffix}.f16.memmap"
        n_samples = cls.infer_n_samples(memmap_path, hidden_dim)
        return cls(cache_dir, key, n_samples, hidden_dim, scope=scope)

    @property
    def expected_bytes(self) -> int:
        return self.n_samples * self.hidden_dim * 2

    def completed_rows(self) -> set[int]:
        return self._load_completed_rows()

    def _load_completed_rows(self) -> set[int]:
        completed: set[int] = set()
        if self._legacy_progress_path.exists():
            data = json.loads(self._legacy_progress_path.read_text(encoding="utf-8"))
            rows = data.get("completed_rows", [])
            if not isinstance(rows, list):
                raise ValueError(f"Invalid progress file: {self._legacy_progress_path}")
            completed.update(int(r) for r in rows)
        if self.progress_path.exists():
            text = self.progress_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row_ids = json.loads(line)
                if not isinstance(row_ids, list):
                    raise ValueError(f"Invalid progress log line in {self.progress_path}: {line!r}")
                completed.update(int(r) for r in row_ids)
        return completed

    def _append_progress(self, row_indices: list[int]) -> None:
        if not row_indices:
            return
        line = json.dumps(row_indices) + "\n"
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def rows_written(self) -> int:
        return len(self._load_completed_rows())

    def _open_memmap(self) -> np.memmap:
        if self._memmap is None:
            if self.memmap_path.exists():
                on_disk_rows = self.infer_n_samples(self.memmap_path, self.hidden_dim)
                if on_disk_rows != self.n_samples:
                    raise MemmapSizeMismatchError(
                        f"Memmap row count mismatch for {self.memmap_path}: "
                        f"file has {on_disk_rows} rows but caller requested "
                        f"n_samples={self.n_samples}. Refusing to resize or "
                        "re-encode silently — use the correct scope/sample parquet "
                        "or a separate train memmap."
                    )
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
        # Append-only (D1): O(batch_size) amortized, not O(N). The memmap is
        # flushed to disk above *before* we acknowledge these rows here, so a
        # kill -9 immediately after this line still leaves the data durable
        # for every row this progress record claims as complete.
        self._append_progress(row_indices)

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
        if self._legacy_progress_path.exists():
            self._legacy_progress_path.unlink()

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
