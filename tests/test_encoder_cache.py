"""Hidden state cache tests (DESIGN.md §14.2)."""

from __future__ import annotations

from pathlib import Path

import torch
from genrec_lite.encode.cache import CacheKeyConfig, HiddenStateCache, compute_cache_key


def test_key_changes_with_model_id() -> None:
    base = {"verbalizer_name": "v1_full", "verbalizer_config": {"max_history": 20}, "max_len": 512}
    k1 = compute_cache_key(CacheKeyConfig(model_id="model-a", **base))
    k2 = compute_cache_key(CacheKeyConfig(model_id="model-b", **base))
    assert k1 != k2


def test_key_changes_with_verbalizer_name() -> None:
    base = {"model_id": "m", "verbalizer_config": {"max_history": 20}, "max_len": 512}
    k1 = compute_cache_key(CacheKeyConfig(verbalizer_name="v1_full", **base))
    k2 = compute_cache_key(CacheKeyConfig(verbalizer_name="v0_ids_only", **base))
    assert k1 != k2


def test_key_changes_with_verbalizer_config() -> None:
    base = {"model_id": "m", "verbalizer_name": "v1_full", "max_len": 512}
    k1 = compute_cache_key(CacheKeyConfig(verbalizer_config={"max_history": 20}, **base))
    k2 = compute_cache_key(CacheKeyConfig(verbalizer_config={"max_history": 10}, **base))
    assert k1 != k2


def test_key_stable_across_process() -> None:
    cfg = CacheKeyConfig(
        model_id="m",
        verbalizer_name="v1_full",
        verbalizer_config={"max_history": 20},
        max_len=512,
    )
    assert compute_cache_key(cfg) == compute_cache_key(cfg)


def test_cache_hit_returns_bit_identical(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "abc", n_samples=2, hidden_dim=4)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    cache.save([1, 2], hidden)
    loaded = cache.load()
    assert torch.equal(loaded.to(torch.float32), hidden)


def test_cache_miss_recomputes(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "key1", n_samples=1, hidden_dim=2)
    assert not cache.exists()
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    cache.save([42], hidden)
    other = HiddenStateCache(tmp_path, "key2", n_samples=1, hidden_dim=2)
    assert not other.exists()
