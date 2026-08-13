"""Baseline recommender tests (DESIGN.md §14.2)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.baselines import build_baseline
from genrec_lite.models.baselines.itemknn import ItemKNNRecommender
from genrec_lite.models.baselines.sasrec import SASRecModel
from genrec_lite.models.baselines.topfreq import GPTopFreqRecommender, PTopFreqRecommender


def test_pop_recommends_most_frequent(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    model = build_baseline("pop")
    model.fit(interactions, items)
    train = interactions.filter(pl.col("split") == 0)
    top_item = train.group_by("item_id").len().sort("len", descending=True)["item_id"][0]
    test_samples = samples.filter(pl.col("split") == 2).head(1)
    scores = model.score_batch(test_samples)
    assert int(np.argmax(scores[0])) == int(top_item)


def test_p_topfreq_prefers_user_history(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    model = PTopFreqRecommender()
    model.fit(interactions, items)
    test_samples = samples.filter(pl.col("split") == 2)
    scores = model.score_batch(test_samples.head(1))
    user_id = int(test_samples["user_id"][0])
    train = interactions.filter((pl.col("split") == 0) & (pl.col("user_id") == user_id))
    if train.height > 0:
        top_user_item = train.group_by("item_id").len().sort("len", descending=True)["item_id"][0]
        assert scores[0, int(top_user_item)] >= scores[0].max() - 1e-6


def test_gp_topfreq_falls_back_to_global_when_short_hist(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    pop = build_baseline("pop")
    pop.fit(interactions, items)
    gp = GPTopFreqRecommender()
    gp.fit(interactions, items)
    short_hist = samples.filter(pl.col("history").list.len() <= 1).head(1)
    if short_hist.height == 0:
        pytest.skip("No short-history sample in fixture")
    pop_scores = pop.score_batch(short_hist)
    gp_scores = gp.score_batch(short_hist)
    assert np.allclose(pop_scores, gp_scores)


def test_itemknn_symmetric_similarity() -> None:
    model = ItemKNNRecommender()
    interactions = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
            "ts": [1, 2, 3, 4],
            "basket_id": [0, 0, 1, 1],
            "rating": [5.0, 5.0, 5.0, 5.0],
            "event_type": [3, 3, 3, 3],
            "split": [0, 0, 0, 0],
        }
    )
    items = pl.DataFrame({"item_id": [0, 1, 2]})
    model.fit(interactions, items)
    assert model._item_similarity(0, 1) == model._item_similarity(1, 0)


def test_sasrec_forward_shape() -> None:
    model = SASRecModel(n_items=5, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=10)
    seq = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0]])
    out = model(seq)
    assert out.shape == (1, 10, 16)


def test_sasrec_causal_mask() -> None:
    model = SASRecModel(n_items=5, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=5)
    model.eval()
    seq = torch.tensor([[1, 2, 3, 4, 0]])
    out1 = model(seq)
    seq_future_changed = torch.tensor([[1, 2, 3, 5, 0]])
    out2 = model(seq_future_changed)
    assert torch.allclose(out1[0, 2], out2[0, 2], atol=1e-5)


def test_pop_is_worst_among_baselines(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)
    baseline_names = ["pop", "p_topfreq", "gp_topfreq", "itemknn", "textknn"]
    ndcg_scores: dict[str, float] = {}
    for name in baseline_names:
        model = build_baseline(name)
        model.fit(interactions, items)
        result = evaluate(
            score_fn=model.score_batch,
            samples=test_samples,
            items=items,
            interactions=interactions,
            ks=(10,),
            slices=("all",),
            method=name,
        )
        ndcg_scores[name] = float(result.iloc[0]["ndcg@10"])
    assert ndcg_scores["pop"] <= min(v for k, v in ndcg_scores.items() if k != "pop")
