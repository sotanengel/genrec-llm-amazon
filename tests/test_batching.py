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


def test_batch_indices_respects_padded_cost_not_sum() -> None:
    """The tensor fed to the model is rectangular: len(batch) * max_len_in_batch,
    not the sum of individual lengths. One long sequence (100) plus two short
    ones (10, 10) sums to 120 (<= 150), so sum-based accounting packs all three
    into a single batch. But the real padded cost of that batch is
    3 * 100 = 300, which blows a 150-token budget. This must fail under the old
    sum-based accounting and pass once batching accounts for padding.
    """
    lengths = [100, 10, 10]
    batches = batch_indices_by_token_budget(lengths, max_batch_tokens=150, batch_size=None)

    for batch in batches:
        max_len = max(lengths[i] for i in batch)
        padded_cost = len(batch) * max_len
        assert padded_cost <= 150, (
            f"batch {batch} has padded cost {padded_cost} > 150 (rectangular "
            f"tensor of shape ({len(batch)}, {max_len}))"
        )

    # The long sequence must not share a batch with anything else.
    assert batches == [[0], [1, 2]]


def test_batch_indices_padded_cost_mixed_lengths_never_exceeds_budget() -> None:
    """Broader mixed-length case: no batch's padded cost may exceed the budget,
    even though several individual sums would still fit under the old
    accounting."""
    lengths = [64, 63, 5, 4, 3, 2, 1]
    max_batch_tokens = 130
    batches = batch_indices_by_token_budget(lengths, max_batch_tokens=max_batch_tokens)

    seen: set[int] = set()
    for batch in batches:
        seen.update(batch)
        max_len = max(lengths[i] for i in batch)
        assert len(batch) * max_len <= max_batch_tokens
    assert seen == set(range(len(lengths)))
