"""Full-catalog ranking evaluation tests (DESIGN.md §14.2)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genrec_lite.eval.runner import evaluate


def test_evaluator_scores_all_items(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)
    n_items = items.height

    def score_fn(batch: pl.DataFrame) -> np.ndarray:
        scores = np.random.default_rng(0).random((batch.height, n_items))
        assert scores.shape[-1] == n_items
        return scores

    result = evaluate(
        score_fn=score_fn,
        samples=test_samples,
        items=items,
        interactions=interactions,
        ks=(10,),
        slices=("all",),
        method="test",
    )
    assert len(result) == 1
    assert result.iloc[0]["method"] == "test"


def test_evaluator_does_not_hide_train_items(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)
    n_items = items.height
    train_items = set(interactions.filter(pl.col("split") == 0)["item_id"].unique().to_list())

    captured_shapes: list[int] = []

    def score_fn(batch: pl.DataFrame) -> np.ndarray:
        scores = np.zeros((batch.height, n_items), dtype=np.float64)
        for item in train_items:
            scores[:, int(item)] = 1.0
        captured_shapes.append(scores.shape[1])
        return scores

    evaluate(
        score_fn=score_fn,
        samples=test_samples,
        items=items,
        interactions=interactions,
        ks=(10,),
        slices=("all",),
    )
    assert captured_shapes[0] == n_items


def test_evaluator_rejects_sampled_ranking(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)

    def bad_score_fn(batch: pl.DataFrame) -> np.ndarray:
        return np.zeros((batch.height, 5))

    with pytest.raises(ValueError, match="Full-catalog ranking required"):
        evaluate(
            score_fn=bad_score_fn,
            samples=test_samples,
            items=items,
            interactions=interactions,
            ks=(10,),
            slices=("all",),
        )
