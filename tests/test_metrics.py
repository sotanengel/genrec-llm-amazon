"""Evaluation metrics tests (DESIGN.md §14.2)."""

from __future__ import annotations

import pytest
import torch

from genrec_lite.eval.metrics import (
    aggregate_metrics,
    avg_popularity_at_k,
    coverage_at_k,
    gini_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)


def test_recall_at_k_hit() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 0.3])
    assert recall_at_k(scores, target=1, k=2) == 1.0


def test_recall_at_k_miss() -> None:
    scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
    assert recall_at_k(scores, target=3, k=2) == 0.0


def test_ndcg_at_k_position_0() -> None:
    scores = torch.tensor([0.9, 0.5, 0.3])
    assert ndcg_at_k(scores, target=0, k=3) == pytest.approx(1.0, abs=1e-4)


def test_ndcg_at_3_single_hit_position_2() -> None:
    # ranking = [item0, TARGET, item2, ...] (target at 0-indexed position 1)
    # NDCG@3 = DCG@3 / IDCG@3
    # DCG@3 = 1 / log2(1 + 2) = 1 / log2(3) ≈ 0.6309
    # IDCG@3 = 1 / log2(1 + 0) = 1.0
    scores = torch.tensor([0.9, 0.6, 0.8, 0.5, 0.4])
    assert ndcg_at_k(scores, target=2, k=3) == pytest.approx(0.6309, abs=1e-4)


def test_mrr_multiple_users() -> None:
    # 3 users with ranks 1, 2, 5 -> MRR = (1 + 0.5 + 0.2) / 3
    scores_list = [
        torch.tensor([0.9, 0.1, 0.2]),
        torch.tensor([0.9, 0.8, 0.7]),
        torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1]),
    ]
    targets = [0, 1, 4]
    ks = [3, 3, 5]
    mrrs = [mrr_at_k(s, t, k) for s, t, k in zip(scores_list, targets, ks, strict=True)]
    assert sum(mrrs) / len(mrrs) == pytest.approx((1.0 + 0.5 + 0.2) / 3, abs=1e-4)


def test_hit_rate_at_k_hit() -> None:
    scores = torch.tensor([0.2, 0.9, 0.1])
    assert hit_rate_at_k(scores, target=1, k=1) == 1.0


def test_coverage_at_k() -> None:
    recommended = [[0, 0, 0], [0, 0, 0]]
    assert coverage_at_k(recommended, n_items=5, k=3) == pytest.approx(1 / 5, abs=1e-4)


def test_gini_uniform_is_zero() -> None:
    recommended = [[0, 1], [0, 1]]
    assert gini_at_k(recommended, n_items=2, k=2) == pytest.approx(0.0, abs=1e-4)


def test_metrics_batch_matches_loop() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    target = 2
    loop = {
        "recall@3": recall_at_k(scores, target, 3),
        "ndcg@3": ndcg_at_k(scores, target, 3),
        "mrr@3": mrr_at_k(scores, target, 3),
    }
    batch = aggregate_metrics([loop])
    assert batch["recall@3"] == loop["recall@3"]
    assert batch["ndcg@3"] == loop["ndcg@3"]
    assert batch["mrr@3"] == loop["mrr@3"]


def test_avg_popularity_and_novelty() -> None:
    popularity = torch.tensor([10.0, 5.0, 1.0, 1.0])
    recommended = [[0, 2], [1, 3]]
    avg_pop = avg_popularity_at_k(recommended, popularity.numpy(), k=2)
    assert avg_pop > 0
    nov = novelty_at_k(recommended, popularity.numpy(), k=2)
    assert nov > 0
