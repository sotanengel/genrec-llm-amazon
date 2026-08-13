"""Baseline model registry."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from genrec_lite.config import SasrecConfig
from genrec_lite.models.baselines.base import BaselineRecommender
from genrec_lite.models.baselines.itemknn import ItemKNNRecommender
from genrec_lite.models.baselines.pop import PopRecommender
from genrec_lite.models.baselines.sasrec import SASRecRecommender
from genrec_lite.models.baselines.textknn import TextKNNRecommender
from genrec_lite.models.baselines.topfreq import GPTopFreqRecommender, PTopFreqRecommender


def build_baseline(
    name: str,
    sasrec_config: SasrecConfig | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> BaselineRecommender:
    """Instantiate a baseline recommender by name."""
    key = name.lower()
    if key == "pop":
        return PopRecommender()
    if key == "p_topfreq":
        return PTopFreqRecommender()
    if key == "gp_topfreq":
        return GPTopFreqRecommender()
    if key == "itemknn":
        return ItemKNNRecommender()
    if key == "sasrec":
        return SASRecRecommender(config=sasrec_config or SasrecConfig())
    if key == "textknn":
        if embed_fn is None:
            from genrec_lite.encode.text_embed import build_text_embed_fn

            embed_fn = build_text_embed_fn()
        return TextKNNRecommender(embed_fn=embed_fn)
    raise ValueError(f"Unknown baseline: {name}")
