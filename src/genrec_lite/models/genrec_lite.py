"""GenRec-lite ranking model (DESIGN.md §4.2)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from genrec_lite.models.ranking_head import RankingHead


class GenRecLite(nn.Module):
    """z = proj(h); s_i = z @ e_i.T (+ optional MLP scorer)."""

    def __init__(
        self,
        d_llm: int,
        d_emb: int,
        n_items: int,
        scorer: Literal["dot", "mlp"] = "dot",
        dropout: float = 0.1,
        item_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head = RankingHead(
            d_llm=d_llm,
            d_emb=d_emb,
            n_items=n_items,
            scorer=scorer,
            dropout=dropout,
            item_init=item_init,
        )

    def score(
        self,
        h: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score items for frozen LLM hidden states."""
        return self.head.score(h, candidate_ids=candidate_ids)

    def forward(
        self,
        h: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.score(h, candidate_ids=candidate_ids)
