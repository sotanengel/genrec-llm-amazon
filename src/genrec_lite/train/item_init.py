"""Item embedding initialization helpers (DESIGN.md §4.3)."""

from __future__ import annotations

from typing import Literal

import polars as pl
import torch
from torch import Tensor

from genrec_lite.encode.text_embed import build_text_embed_fn
from genrec_lite.models.baselines.textknn import _default_item_text
from genrec_lite.models.genrec_lite import GenRecLite

ItemInitMode = Literal["random", "text", "text_frozen"]


def build_item_init_matrix(
    items: pl.DataFrame,
    mode: ItemInitMode,
    embed_model: str,
) -> tuple[Tensor | None, bool]:
    """Return optional init matrix and whether item embeddings should be frozen."""
    if mode == "random":
        return None, False

    texts = [_default_item_text(row) for row in items.iter_rows(named=True)]
    embed_fn = build_text_embed_fn(model_name=embed_model)
    matrix = torch.from_numpy(embed_fn(texts).astype("float32"))
    freeze = mode == "text_frozen"
    return matrix, freeze


def apply_item_init_freeze(model: GenRecLite, freeze: bool) -> None:
    """Optionally freeze item embedding weights after initialization."""
    if freeze:
        model.head.item_emb.weight.requires_grad_(False)
