"""Baseline recommender protocol (DESIGN.md §6)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import polars as pl


class BaselineRecommender(Protocol):
    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None: ...

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray: ...

    def name(self) -> str: ...
