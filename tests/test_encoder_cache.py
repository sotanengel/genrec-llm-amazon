"""Hidden state cache tests (DESIGN.md §14.2)."""

from __future__ import annotations

from pathlib import Path

import torch

from genrec_lite.encode.cache import CacheKeyConfig, HiddenStateCache, compute_cache_key
from genrec_lite.encode.prefill import ENCODER_VERSION


def _base_key_config(**overrides: object) -> CacheKeyConfig:
    defaults: dict[str, object] = {
        "model_id": "m",
        "revision": "abc123",
        "verbalizer_name": "v1_full",
        "verbalizer_config": {"max_history": 20},
        "max_len": 512,
        "dtype": "bfloat16",
        "quantize": None,
        "pooling": "last",
        "attn_implementation": "auto",
        "deterministic": False,
        "encoder_version": ENCODER_VERSION,
    }
    defaults.update(overrides)
    return CacheKeyConfig(**defaults)  # type: ignore[arg-type]


def test_key_changes_with_model_id() -> None:
    k1 = compute_cache_key(_base_key_config(model_id="model-a"))
    k2 = compute_cache_key(_base_key_config(model_id="model-b"))
    assert k1 != k2


def test_key_changes_with_verbalizer_name() -> None:
    k1 = compute_cache_key(_base_key_config(verbalizer_name="v1_full"))
    k2 = compute_cache_key(_base_key_config(verbalizer_name="v0_ids_only"))
    assert k1 != k2


def test_key_changes_with_verbalizer_config() -> None:
    k1 = compute_cache_key(_base_key_config(verbalizer_config={"max_history": 20}))
    k2 = compute_cache_key(_base_key_config(verbalizer_config={"max_history": 10}))
    assert k1 != k2


def test_key_changes_with_dtype_and_quantize() -> None:
    k1 = compute_cache_key(_base_key_config(dtype="bfloat16", quantize=None))
    k2 = compute_cache_key(_base_key_config(dtype="bfloat16", quantize="nf4"))
    assert k1 != k2


def test_key_changes_with_encoder_version() -> None:
    k1 = compute_cache_key(_base_key_config(encoder_version=1))
    k2 = compute_cache_key(_base_key_config(encoder_version=2))
    assert k1 != k2


def test_key_stable_across_process() -> None:
    cfg = _base_key_config()
    assert compute_cache_key(cfg) == compute_cache_key(cfg)


def test_cache_hit_returns_bit_identical(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "abc", n_samples=2, hidden_dim=4)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    cache.save([1, 2], hidden)
    loaded = cache.load()
    assert torch.equal(loaded.to(torch.float32), hidden)


def test_write_rows_accepts_bfloat16(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "bf16", n_samples=1, hidden_dim=2)
    hidden = torch.tensor([[1.5, 2.5]], dtype=torch.bfloat16)
    cache.write_rows([0], hidden)
    cache.finalize([1])
    loaded = cache.load()
    assert loaded.shape == (1, 2)
    assert torch.allclose(loaded.to(torch.float32), hidden.to(torch.float32), atol=1e-3)


def test_write_rows_and_finalize_roundtrip(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "stream", n_samples=3, hidden_dim=2)
    cache.write_rows([2], torch.tensor([[1.0, 2.0]]))
    cache.write_rows([0, 1], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    cache.finalize([10, 11, 12], meta={"model_id": "m", "revision": "abc"})
    loaded = cache.load()
    expected = torch.tensor([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
    assert torch.equal(loaded.to(torch.float32), expected)
    assert cache.meta_path.exists()


def test_resume_after_partial_write(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "resume", n_samples=2, hidden_dim=2)
    cache.write_rows([0], torch.tensor([[1.0, 2.0]]))
    assert cache.rows_written() == 1
    assert not cache.exists()
    cache.write_rows([1], torch.tensor([[3.0, 4.0]]))
    cache.finalize([1, 2])
    assert cache.exists()


def test_cache_miss_recomputes(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "key1", n_samples=1, hidden_dim=2)
    assert not cache.exists()
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    cache.save([42], hidden)
    other = HiddenStateCache(tmp_path, "key2", n_samples=1, hidden_dim=2)
    assert not other.exists()
