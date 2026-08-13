"""PrefillEncoder.from_config tests (issue #7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from genrec_lite.config import LLMConfig
from genrec_lite.encode.prefill import PrefillEncoder


def test_from_config_passes_revision_to_from_pretrained() -> None:
    cfg = LLMConfig(
        model_id="sshleifer/tiny-gpt2",
        revision="deadbeef0123456789abcdef0123456789abcdef",
        license="Apache-2.0",
        dtype="float32",
        device="cpu",
    )
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"

    mock_model = MagicMock()
    mock_model.parameters.return_value = []

    def fake_tokenizer_from_pretrained(_model_id: str, **kwargs: object) -> MagicMock:
        tokenizer_calls.append(dict(kwargs))
        return mock_tokenizer

    def fake_model_from_pretrained(_model_id: str, **kwargs: object) -> MagicMock:
        model_calls.append(dict(kwargs))
        return mock_model

    with (
        patch(
            "genrec_lite.encode.prefill.AutoTokenizer.from_pretrained",
            side_effect=fake_tokenizer_from_pretrained,
        ),
        patch(
            "genrec_lite.encode.prefill.AutoModel.from_pretrained",
            side_effect=fake_model_from_pretrained,
        ),
    ):
        PrefillEncoder.from_config(cfg)

    assert tokenizer_calls[0]["revision"] == cfg.revision
    assert model_calls[0]["revision"] == cfg.revision


def _mocked_encoder(**kwargs: object) -> PrefillEncoder:
    """Build a PrefillEncoder against mocked tokenizer/model so tests don't
    need network access or a real GPU. device="cuda" is safe here because
    the model/tokenizer are mocks: no real CUDA call is ever made."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"

    mock_model = MagicMock()
    mock_model.parameters.return_value = []

    with (
        patch(
            "genrec_lite.encode.prefill.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "genrec_lite.encode.prefill.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
    ):
        return PrefillEncoder(model_id="sshleifer/tiny-gpt2", dtype="float32", **kwargs)  # type: ignore[arg-type]


def test_deterministic_flag_is_restored_by_close() -> None:
    """torch.use_deterministic_algorithms is a process-global flag. A
    PrefillEncoder that turns it on must be able to turn it back off via
    close(), so a later non-deterministic encoder in the same process does
    not silently inherit it."""
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        encoder = _mocked_encoder(device="cuda", deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        encoder.close()
        assert torch.are_deterministic_algorithms_enabled() == prior
    finally:
        torch.use_deterministic_algorithms(prior)


def test_deterministic_flag_is_restored_by_context_manager() -> None:
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        with _mocked_encoder(device="cuda", deterministic=True) as encoder:
            assert torch.are_deterministic_algorithms_enabled() is True
            assert encoder is not None
        assert torch.are_deterministic_algorithms_enabled() == prior
    finally:
        torch.use_deterministic_algorithms(prior)


def test_non_deterministic_encoder_does_not_touch_global_flag() -> None:
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        encoder = _mocked_encoder(device="cuda", deterministic=False)
        assert torch.are_deterministic_algorithms_enabled() == prior
        encoder.close()  # must be a safe no-op
        assert torch.are_deterministic_algorithms_enabled() == prior
    finally:
        torch.use_deterministic_algorithms(prior)
