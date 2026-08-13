"""Evaluation slice tests (DESIGN.md §14.2)."""

from __future__ import annotations

import polars as pl
from genrec_lite.eval.runner import evaluate
from genrec_lite.eval.slices import all_slice_names, assign_slice, filter_slice


def test_repeat_flag_correctness(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    enriched = assign_slice(samples, items, interactions)
    repeat_rows = enriched.filter(pl.col("is_repeat"))
    for row in repeat_rows.iter_rows(named=True):
        assert row["target_item"] in row["history"]


def test_cold_flag_uses_threshold(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    enriched = assign_slice(samples, items, interactions, cold_threshold=5)
    cold = filter_slice(enriched, "cold", cold_threshold=5)
    for row in cold.iter_rows(named=True):
        assert row["target_n_train_inter"] < 5


def test_all_slice_counts_sum_to_total(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)
    enriched = assign_slice(test_samples, items, interactions)
    repeat = filter_slice(enriched, "repeat").height
    explore = filter_slice(enriched, "explore").height
    assert repeat + explore == enriched.height


def test_slice_metrics_never_nan_when_nonempty(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)

    def score_fn(batch: pl.DataFrame) -> object:
        import numpy as np

        n_items = items.height
        rng = np.random.default_rng(0)
        return rng.random((batch.height, n_items))

    result = evaluate(
        score_fn=score_fn,
        samples=test_samples,
        items=items,
        interactions=interactions,
        ks=(10,),
        slices=("all", "repeat", "explore"),
    )
    numeric_cols = [c for c in result.columns if c not in {"method", "slice", "n_samples"}]
    for col in numeric_cols:
        values = result[col].dropna().tolist()
        if values:
            assert all(not (isinstance(v, float) and v != v) for v in values)


def test_pop_decile_slices_filter(mini_bundle: tuple) -> None:
    interactions, items, users, samples = mini_bundle
    test_samples = samples.filter(pl.col("split") == 2)
    enriched = assign_slice(test_samples, items, interactions)
    decile_names = [s for s in all_slice_names() if s.startswith("pop_decile_")]
    assert len(decile_names) == 10
    total = 0
    for name in decile_names:
        subset = filter_slice(enriched, name)
        total += subset.height
        for row in subset.iter_rows(named=True):
            assert row["pop_decile"] == name
    assert total == enriched.height
