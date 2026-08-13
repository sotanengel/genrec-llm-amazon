"""Token batching tests (issue #9)."""

from __future__ import annotations

from genrec_lite.encode.batching import batch_indices_by_token_budget


def test_batch_indices_respects_max_batch_tokens() -> None:
    lengths = [10, 20, 30, 40]
    batches = batch_indices_by_token_budget(lengths, max_batch_tokens=50, batch_size=None)
    for batch in batches:
        total = sum(lengths[i] for i in batch)
        assert total <= 50


def test_batch_indices_sorts_longest_first() -> None:
    lengths = [5, 50, 10]
    batches = batch_indices_by_token_budget(lengths, max_batch_tokens=100, batch_size=3)
    assert batches[0][0] == 1
