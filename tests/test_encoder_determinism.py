"""Encoder determinism tests (DESIGN.md §14.2)."""

from __future__ import annotations

import torch

from genrec_lite.encode.prefill import PrefillEncoder


def test_same_input_same_output(tiny_model_id: str) -> None:
    encoder = PrefillEncoder(
        model_id=tiny_model_id,
        dtype="float32",
        pooling="last",
        max_len=32,
        device="cpu",
    )
    text = "deterministic encoding check"
    out1 = encoder.encode_batch([text])
    out2 = encoder.encode_batch([text])
    assert torch.allclose(out1, out2, rtol=0, atol=1e-5)


def test_no_grad_by_default(tiny_model_id: str) -> None:
    encoder = PrefillEncoder(
        model_id=tiny_model_id,
        dtype="float32",
        pooling="last",
        max_len=32,
        device="cpu",
    )
    for param in encoder.model.parameters():
        assert not param.requires_grad
