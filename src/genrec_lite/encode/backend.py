"""Device/dtype/attention/quantization resolvers for PrefillEncoder (issue #7)."""

from __future__ import annotations

import importlib.util
from typing import Literal

import torch

AttnImplementation = Literal["auto", "sdpa", "flash_attention_2", "eager"]


def resolve_device(device: str | None = None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    return torch.float32


def supports_bf16(device: str | None = None) -> bool:
    resolved = resolve_device(device)
    if resolved == "cpu":
        return False
    return bool(torch.cuda.is_bf16_supported())


def supports_fa2(device: str | None = None) -> bool:
    resolved = resolve_device(device)
    if resolved == "cpu":
        return False
    if importlib.util.find_spec("flash_attn") is None:
        return False
    capability = torch.cuda.get_device_capability()
    return capability >= (8, 0)


def resolve_attn_impl(
    attn_implementation: AttnImplementation | str,
    dtype: str,
    device: str | None = None,
) -> str:
    resolved_device = resolve_device(device)

    if resolved_device == "cpu":
        return "eager"

    if attn_implementation in {"sdpa", "eager", "flash_attention_2"}:
        if attn_implementation == "flash_attention_2" and not supports_fa2(resolved_device):
            return "sdpa"
        return attn_implementation

    resolved_dtype = resolve_dtype(dtype)
    if supports_fa2(resolved_device) and resolved_dtype in {torch.bfloat16, torch.float16}:
        return "flash_attention_2"
    return "sdpa"


def build_quant_config(bnb_compute_dtype: str = "bfloat16") -> object:
    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_dtype(bnb_compute_dtype)
    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
