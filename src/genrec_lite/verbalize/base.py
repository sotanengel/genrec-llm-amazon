"""Verbalizer protocol (DESIGN.md §4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class Sample:
    user_id: int
    cutoff_ts: int
    target_item: int
    history: list[int]
    is_repeat: bool
    target_is_cold: bool


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int
    tokenizer_name: str
    # Pinned revision for `tokenizer_name` (DESIGN.md §2.4.4). `None` when the
    # tokenizer isn't pinned to a specific revision -- see
    # `genrec_lite.cli._resolve_tokenizer_revision` for how this is derived.
    revision: str | None = None


class Verbalizer(Protocol):
    def render(
        self,
        sample: Sample,
        items: pl.DataFrame,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
        budget: TokenBudget,
    ) -> str: ...

    def name(self) -> str: ...
