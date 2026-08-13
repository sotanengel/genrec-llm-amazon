"""Token-length batching helpers for encode run (issue #9)."""

from __future__ import annotations


def batch_indices_by_token_budget(
    lengths: list[int],
    max_batch_tokens: int,
    batch_size: int | None = None,
) -> list[list[int]]:
    """Group row indices into batches sorted by descending length.

    The budget is enforced against the *padded* cost actually fed to the
    model: a rectangular tensor of shape (len(batch), max_len_in_batch), i.e.
    `len(batch) * max_len_in_batch` — not the sum of the individual sequence
    lengths (D2). Summing under-counts the real cost because every row in a
    batch is padded up to the batch's longest row.

    The descending sort is deliberate and preserved: because `order` is
    sorted by length descending, the first element placed into a batch is
    always that batch's longest member, so the padded cost of a batch is
    simply `len(batch) * lengths[batch[0]]`. This also means an
    over-budget batch is detected on the very first (longest) batch rather
    than ~90% of the way through a multi-hour run.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    # Longest sequence length among the rows already placed into `current`.
    # Always `lengths[current[0]]` because `order` is descending.
    batch_max_len = 0

    for idx in order:
        seq_len = lengths[idx]
        if batch_size is not None and len(current) >= batch_size:
            batches.append(current)
            current = []
            batch_max_len = 0
        if current and (len(current) + 1) * batch_max_len > max_batch_tokens:
            batches.append(current)
            current = []
            batch_max_len = 0
        current.append(idx)
        if len(current) == 1:
            batch_max_len = seq_len

    if current:
        batches.append(current)
    return batches
