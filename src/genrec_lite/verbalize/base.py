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
