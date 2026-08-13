"""Tests for encode/backend.py resolver helpers (issue #7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genrec_lite.config import LLMConfig
from genrec_lite.encode.backend import resolve_attn_impl


def test_resolve_attn_impl_defaults_to_sdpa_when_flash_attn_missing() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        assert resolve_attn_impl("auto", "bfloat16", device="cuda") == "sdpa"


def test_resolve_attn_impl_uses_fa2_when_available_and_capable() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (8, 6)
    with (
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch("genrec_lite.encode.backend.torch", mock_torch),
    ):
        assert resolve_attn_impl("auto", "bfloat16", device="cuda") == "flash_attention_2"


def test_resolve_attn_impl_cpu_uses_eager() -> None:
    assert resolve_attn_impl("auto", "bfloat16", device="cpu") == "eager"


def test_llm_config_rejects_floating_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        LLMConfig(
            model_id="Qwen/Qwen3-1.7B-Base",
            revision="main",
            license="Apache-2.0",
        )
