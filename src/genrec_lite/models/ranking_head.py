"""Catalog-constrained ranking head (DESIGN.md §4.1)."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn


class RankingHead(nn.Module):
    """Project LLM hidden states and score items in the catalog."""

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
        self.d_emb = d_emb
        self.n_items = n_items
        self.scorer_type = scorer

        self.proj = nn.Linear(d_llm, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.item_emb = nn.Embedding(n_items, d_emb)
        self.item_init_proj: nn.Linear | None = None

        if item_init is not None:
            init_dim = item_init.shape[1]
            if init_dim != d_emb:
                self.item_init_proj = nn.Linear(init_dim, d_emb, bias=False)
            self._initialize_item_embeddings(item_init)

        if scorer == "mlp":
            self.scorer_mlp: nn.Module = nn.Sequential(
                nn.Linear(d_emb, d_emb),
                nn.ReLU(),
                nn.Linear(d_emb, d_emb),
            )
        else:
            self.scorer_mlp = nn.Identity()

    def _initialize_item_embeddings(self, item_init: torch.Tensor) -> None:
        if item_init.shape[0] != self.n_items:
            msg = f"item_init rows {item_init.shape[0]} != n_items {self.n_items}"
            raise ValueError(msg)
        init_dim = item_init.shape[1]
        with torch.no_grad():
            if init_dim == self.d_emb:
                self.item_emb.weight.copy_(item_init)
                return
            if self.item_init_proj is None:
                msg = f"item_init dim {init_dim} != d_emb {self.d_emb} but no projection layer"
                raise ValueError(msg)
            projected = self.item_init_proj(item_init)
            self.item_emb.weight.copy_(projected)

    def _user_repr(self, h: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.proj(h))
        return cast(torch.Tensor, self.scorer_mlp(z))

    def score(
        self,
        h: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return catalog or candidate-set scores for batch of hidden states."""
        z = self._user_repr(h)
        if candidate_ids is None:
            return z @ self.item_emb.weight.T

        item_vecs = self.item_emb(candidate_ids)
        if item_vecs.dim() == 2:
            return cast(torch.Tensor, z @ item_vecs.T)
        return cast(torch.Tensor, (z.unsqueeze(1) * item_vecs).sum(dim=-1))
