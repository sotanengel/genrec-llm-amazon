"""Baseline model registry."""

from __future__ import annotations

from genrec_lite.config import SasrecConfig
from genrec_lite.models.baselines.base import BaselineRecommender
from genrec_lite.models.baselines.itemknn import ItemKNNRecommender
from genrec_lite.models.baselines.pop import PopRecommender
from genrec_lite.models.baselines.sasrec import SASRecRecommender
from genrec_lite.models.baselines.textknn import TextKNNRecommender
from genrec_lite.models.baselines.topfreq import GPTopFreqRecommender, PTopFreqRecommender


def build_baseline(name: str, sasrec_config: SasrecConfig | None = None) -> BaselineRecommender:
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
        return TextKNNRecommender()
    raise ValueError(f"Unknown baseline: {name}")
