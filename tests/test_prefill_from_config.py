"""PrefillEncoder.from_config tests (issue #7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
