"""Prefill-only LLM encoder (DESIGN.md §4.2)."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

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
    ) -> None:
        self.model_id = model_id
        self.pooling = pooling
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: dict[str, object] = {"torch_dtype": torch_dtype}
        if quantize == "nf4":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

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
