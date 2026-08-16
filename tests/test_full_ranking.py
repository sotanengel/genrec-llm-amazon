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


def test_evaluator_scores_samples_in_bounded_batches(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    del users
    eval_samples = samples.head(5)
    observed_batch_sizes: list[int] = []

    def score_fn(batch: pl.DataFrame) -> np.ndarray:
        observed_batch_sizes.append(batch.height)
        sample_ids = np.asarray(batch["sample_id"].to_list(), dtype=np.float64)
        item_ids = np.arange(items.height, dtype=np.float64)
        return sample_ids[:, None] * 0.01 + item_ids[None, :]

    evaluate(
        score_fn=score_fn,
        samples=eval_samples,
        items=items,
        interactions=interactions,
        ks=(2,),
        slices=("all",),
        eval_batch_size=2,
    )

    assert observed_batch_sizes == [2, 2, 1]


def test_batched_evaluation_preserves_global_metrics(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    del users
    eval_samples = samples.head(5)

    def score_fn(batch: pl.DataFrame) -> np.ndarray:
        sample_ids = np.asarray(batch["sample_id"].to_list(), dtype=np.float64)
        item_ids = np.arange(items.height, dtype=np.float64)
        return -np.abs(sample_ids[:, None] % items.height - item_ids[None, :])

    unbatched = evaluate(
        score_fn=score_fn,
        samples=eval_samples,
        items=items,
        interactions=interactions,
        ks=(1, 3),
        slices=("all",),
        eval_batch_size=eval_samples.height,
    )
    batched = evaluate(
        score_fn=score_fn,
        samples=eval_samples,
        items=items,
        interactions=interactions,
        ks=(1, 3),
        slices=("all",),
        eval_batch_size=2,
    )

    pd_columns = [
        "recall@1",
        "ndcg@3",
        "mrr@3",
        "coverage@3",
        "gini@3",
        "avg_popularity@3",
        "novelty@3",
    ]
    np.testing.assert_allclose(
        batched[pd_columns].to_numpy(),
        unbatched[pd_columns].to_numpy(),
    )


def test_evaluator_rejects_non_positive_batch_size(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    del users
    with pytest.raises(ValueError, match="eval_batch_size"):
        evaluate(
            score_fn=lambda batch: np.zeros((batch.height, items.height)),
            samples=samples.head(1),
            items=items,
            interactions=interactions,
            slices=("all",),
            eval_batch_size=0,
        )
