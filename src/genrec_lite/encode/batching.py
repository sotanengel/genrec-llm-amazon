"""Token-length batching helpers for encode run (issue #9)."""

from __future__ import annotations


def batch_indices_by_token_budget(
    lengths: list[int],
    max_batch_tokens: int,
    batch_size: int | None = None,
) -> list[list[int]]:
    """Group row indices into batches sorted by descending length."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for idx in order:
        seq_len = lengths[idx]
        if batch_size is not None and len(current) >= batch_size:
            batches.append(current)
            current = []
            current_tokens = 0
        if current and current_tokens + seq_len > max_batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += seq_len

    if current:
        batches.append(current)
    return batches
