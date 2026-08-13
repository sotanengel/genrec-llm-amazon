"""Prefill-only LLM encoder (DESIGN.md §4.2)."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from genrec_lite.config import LLMConfig
from genrec_lite.encode.backend import (
    build_quant_config,
    resolve_attn_impl,
    resolve_device,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

PoolingMode = Literal["last", "mean", "eos"]


class PrefillEncoder:
    """LLM prefill-only encoder returning pooled hidden states."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        pooling: PoolingMode = "last",
        max_len: int = 512,
        quantize: str | None = None,
        device: str | None = None,
        *,
        revision: str | None = None,
        attn_implementation: str = "auto",
        low_cpu_mem_usage: bool = True,
        bnb_compute_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.pooling = pooling
        self.max_len = max_len
        self.device = resolve_device(device)
        self.revision = revision

        torch_dtype = resolve_dtype(dtype)
        from_pretrained_common: dict[str, object] = {}
        if revision is not None:
            from_pretrained_common["revision"] = revision
        if trust_remote_code:
            from_pretrained_common["trust_remote_code"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **from_pretrained_common)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        attn_impl = resolve_attn_impl(attn_implementation, dtype, self.device)

        load_kwargs: dict[str, object] = {
            "dtype": torch_dtype,
            "attn_implementation": attn_impl,
            **from_pretrained_common,
        }

        uses_quant = quantize == "nf4"
        if uses_quant:
            load_kwargs["quantization_config"] = build_quant_config(bnb_compute_dtype)
            load_kwargs["device_map"] = {"": 0} if self.device == "cuda" else "auto"
            load_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        elif low_cpu_mem_usage:
            load_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if not uses_quant:
            self.model.to(self.device)

    @classmethod
    def from_config(cls, cfg: LLMConfig, device: str | None = None) -> PrefillEncoder:
        return cls(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            pooling=cfg.pooling,
            max_len=cfg.max_len,
            quantize=cfg.quantize,
            device=device or cfg.device,
            revision=cfg.revision,
            attn_implementation=cfg.attn_implementation,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            bnb_compute_dtype=cfg.bnb_compute_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )

    def _pool(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        seq_len = attention_mask.shape[1]
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        last_indices = (positions * attention_mask.long()).argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        if self.pooling == "eos":
            return hidden[batch_idx, last_indices]
        return hidden[batch_idx, last_indices]

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> Tensor:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=False)
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden, encoded["attention_mask"])
        return pooled.cpu()
