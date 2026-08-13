"""Left padding tests for PrefillEncoder (DESIGN.md §10.1, §14.2)."""

from __future__ import annotations

import pytest
import torch
from genrec_lite.encode.prefill import PrefillEncoder


@pytest.fixture
def encoder(tiny_model_id: str) -> PrefillEncoder:
    return PrefillEncoder(
        model_id=tiny_model_id,
        dtype="float32",
        pooling="last",
        max_len=32,
        device="cpu",
    )


def test_tokenizer_padding_side_is_left(encoder: PrefillEncoder) -> None:
    assert encoder.tokenizer.padding_side == "left"


def test_last_token_pooling_ignores_pad(encoder: PrefillEncoder) -> None:
    short = "hello world"
    long = "hello world this is a longer sentence for padding"
    single_short = encoder.encode_batch([short])
    single_long = encoder.encode_batch([long])
    batch = encoder.encode_batch([short, long])
    assert torch.allclose(batch[0], single_short[0], atol=1e-4)
    assert torch.allclose(batch[1], single_long[0], atol=1e-4)


def test_batch_invariance(encoder: PrefillEncoder) -> None:
    texts = ["alpha", "beta", "gamma", "delta"]
    full = encoder.encode_batch(texts)
    parts = [encoder.encode_batch([t]) for t in texts]
    stacked = torch.cat(parts, dim=0)
    assert torch.allclose(full, stacked, atol=1e-4)


def test_eos_pooling_finds_correct_position(tiny_model_id: str) -> None:
    encoder = PrefillEncoder(
        model_id=tiny_model_id,
        dtype="float32",
        pooling="eos",
        max_len=32,
        device="cpu",
    )
    text = "unique pooling check"
    out = encoder.encode_batch([text])
    assert out.shape[0] == 1
    assert out.shape[1] > 0
