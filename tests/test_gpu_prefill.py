"""GPU prefill smoke tests (issue #11). Run via `make test-gpu` in WSL only."""

from __future__ import annotations

import pytest
import torch

from genrec_lite.encode.prefill import PrefillEncoder


@pytest.mark.gpu
def test_gpu_prefill_encode_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    encoder = PrefillEncoder(
        model_id="sshleifer/tiny-gpt2",
        dtype="float32",
        max_len=32,
        device="cuda",
    )
    out = encoder.encode_batch(["gpu smoke test"])
    assert out.shape[0] == 1
    assert out.shape[1] > 0
