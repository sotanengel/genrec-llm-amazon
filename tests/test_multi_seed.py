"""Multi-seed metrics aggregation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from genrec_lite.report.build import aggregate_metrics_across_seeds


def test_aggregate_metrics_across_seeds_mean_std() -> None:
    # Two seeds, same method/slice: ndcg@10 values 0.10 and 0.20 -> mean=0.15, std=0.05
    metrics = pd.DataFrame(
        [
            {"method": "pop", "slice": "all", "n_samples": 10, "seed": 0, "ndcg@10": 0.10},
            {"method": "pop", "slice": "all", "n_samples": 10, "seed": 1, "ndcg@10": 0.20},
        ]
    )
    summary = aggregate_metrics_across_seeds(metrics)
    row = summary[(summary["method"] == "pop") & (summary["slice"] == "all")].iloc[0]
    assert row["ndcg@10"] == pytest.approx(0.15)
    assert row["ndcg@10_std"] == pytest.approx(0.05)
