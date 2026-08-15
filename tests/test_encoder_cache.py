"""Hidden state cache tests (DESIGN.md §14.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from genrec_lite.encode.cache import (
    CacheKeyConfig,
    HiddenStateCache,
    MemmapSizeMismatchError,
    compute_cache_key,
)
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


def test_encoder_version_does_not_silently_default_to_stale_value() -> None:
    """A caller that forgets to pass encoder_version must not silently get a
    key for a different encoder generation. The default must track the real
    ENCODER_VERSION, not a hardcoded historical value."""
    cfg = CacheKeyConfig(
        model_id="m",
        revision="abc123",
        verbalizer_name="v1_full",
        verbalizer_config={"max_history": 20},
        max_len=512,
    )
    assert cfg.encoder_version == ENCODER_VERSION


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


def test_write_rows_appends_progress_without_rewriting_full_history(tmp_path: Path) -> None:
    """write_rows must be O(1) amortized per batch: each call appends one
    record instead of reading + re-serializing the whole completed-rows set.
    A single JSON-object-per-file format (the old design) collapses to one
    line no matter how many calls are made; an append-only log grows one
    line per call."""
    cache = HiddenStateCache(tmp_path, "append", n_samples=4, hidden_dim=1)
    cache.write_rows([0], torch.tensor([[1.0]]))
    cache.write_rows([1], torch.tensor([[2.0]]))
    cache.write_rows([2], torch.tensor([[3.0]]))

    lines = [line for line in cache.progress_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3, (
        "progress file should gain one appended record per write_rows call, "
        f"got {len(lines)} lines: {lines!r}"
    )
    assert cache.rows_written() == 3
    assert cache.completed_rows() == {0, 1, 2}


def test_resume_after_crash_produces_byte_identical_memmap(tmp_path: Path) -> None:
    """An interrupted-then-resumed run (object dropped mid-way, reopened, and
    finished) must produce a byte-identical memmap to an uninterrupted run,
    and the file must be exactly n_samples * hidden_dim * 2 bytes."""
    n_samples, hidden_dim = 6, 3
    hidden_full = torch.arange(n_samples * hidden_dim, dtype=torch.float32).reshape(
        n_samples, hidden_dim
    )
    sample_ids = list(range(100, 100 + n_samples))

    uninterrupted_dir = tmp_path / "uninterrupted"
    cache_a = HiddenStateCache(uninterrupted_dir, "k", n_samples=n_samples, hidden_dim=hidden_dim)
    cache_a.write_rows(list(range(n_samples)), hidden_full)
    cache_a.finalize(sample_ids)

    resumed_dir = tmp_path / "resumed"
    half = n_samples // 2
    cache_b1 = HiddenStateCache(resumed_dir, "k", n_samples=n_samples, hidden_dim=hidden_dim)
    cache_b1.write_rows(list(range(0, half)), hidden_full[:half])
    del cache_b1  # simulate a crash: no finalize, no clean shutdown

    cache_b2 = HiddenStateCache(resumed_dir, "k", n_samples=n_samples, hidden_dim=hidden_dim)
    assert cache_b2.rows_written() == half
    cache_b2.write_rows(list(range(half, n_samples)), hidden_full[half:])
    cache_b2.finalize(sample_ids)

    bytes_a = (uninterrupted_dir / "k.f16.memmap").read_bytes()
    bytes_b = (resumed_dir / "k.f16.memmap").read_bytes()
    expected_size = n_samples * hidden_dim * 2
    assert len(bytes_a) == expected_size
    assert len(bytes_b) == expected_size
    assert bytes_a == bytes_b


def test_legacy_progress_json_is_honoured_on_resume(tmp_path: Path) -> None:
    """A pre-upgrade `*.progress.json` (single JSON object, {"completed_rows":
    [...]}) must still be read on resume rather than silently restarting a
    partially finished multi-hour job."""
    cache = HiddenStateCache(tmp_path, "legacy", n_samples=2, hidden_dim=2)
    legacy_path = tmp_path / "legacy.progress.json"
    legacy_path.write_text(json.dumps({"completed_rows": [0]}), encoding="utf-8")

    assert cache.completed_rows() == {0}
    assert cache.rows_written() == 1

    # Resume: only the remaining row needs to be written.
    cache.write_rows([1], torch.tensor([[3.0, 4.0]]))
    assert cache.rows_written() == 2
    cache.finalize([10, 11])
    assert cache.exists()


def test_legacy_and_new_progress_formats_are_merged(tmp_path: Path) -> None:
    """If both an old-format file and new-format appends exist (e.g. a run
    upgraded mid-flight), completed rows from both must be honoured."""
    cache = HiddenStateCache(tmp_path, "mixed", n_samples=3, hidden_dim=1)
    legacy_path = tmp_path / "mixed.progress.json"
    legacy_path.write_text(json.dumps({"completed_rows": [0]}), encoding="utf-8")

    cache.write_rows([1], torch.tensor([[2.0]]))
    assert cache.completed_rows() == {0, 1}

    cache.write_rows([2], torch.tensor([[3.0]]))
    cache.finalize([10, 11, 12])
    assert cache.exists()
    # finalize() should clean up both progress files.
    assert not legacy_path.exists()
    assert not cache.progress_path.exists()


def test_cache_scope_eval_uses_legacy_paths(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "abc", n_samples=1, hidden_dim=2, scope="eval")
    assert cache.memmap_path.name == "abc.f16.memmap"
    assert cache.index_path.name == "abc.index.parquet"


def test_cache_scope_train_uses_suffix_paths(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "abc", n_samples=1, hidden_dim=2, scope="train")
    assert cache.memmap_path.name == "abc.train.f16.memmap"
    assert cache.index_path.name == "abc.train.index.parquet"


def test_cache_load_infers_n_samples_from_file_size(tmp_path: Path) -> None:
    n_samples, hidden_dim = 4, 3
    cache = HiddenStateCache(
        tmp_path, "k", n_samples=n_samples, hidden_dim=hidden_dim, scope="eval"
    )
    hidden = torch.arange(n_samples * hidden_dim, dtype=torch.float32).reshape(
        n_samples, hidden_dim
    )
    cache.save(list(range(n_samples)), hidden)

    opened = HiddenStateCache.open_existing(tmp_path, "k", hidden_dim, scope="eval")
    assert opened.n_samples == n_samples
    assert torch.equal(opened.load().to(torch.float32), hidden)


def test_cache_rejects_size_mismatch_without_reencode(tmp_path: Path) -> None:
    cache = HiddenStateCache(tmp_path, "k", n_samples=2, hidden_dim=2, scope="eval")
    cache.save([1, 2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    bad = HiddenStateCache(tmp_path, "k", n_samples=5, hidden_dim=2, scope="eval")
    with pytest.raises(MemmapSizeMismatchError, match="row count mismatch"):
        bad.write_rows([0], torch.tensor([[1.0, 2.0]]))


def test_eval_and_train_memmaps_are_independent(tmp_path: Path) -> None:
    eval_cache = HiddenStateCache(tmp_path, "k", n_samples=1, hidden_dim=2, scope="eval")
    train_cache = HiddenStateCache(tmp_path, "k", n_samples=2, hidden_dim=2, scope="train")
    eval_cache.save([10], torch.tensor([[1.0, 2.0]]))
    train_cache.save([20, 21], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert eval_cache.memmap_path.exists()
    assert train_cache.memmap_path.exists()
    assert eval_cache.memmap_path != train_cache.memmap_path
