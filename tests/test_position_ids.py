"""position_ids computation tests (issue #9, DESIGN.md §14.2)."""

from __future__ import annotations

import torch

from genrec_lite.encode.prefill import ENCODER_VERSION, compute_position_ids


def test_compute_position_ids_left_padded_batch() -> None:
    # Left padding: pad tokens get position 0, real tokens get 0..L-1
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    )
    position_ids = compute_position_ids(attention_mask)
    expected = torch.tensor(
        [
            [0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3],
        ]
    )
    assert torch.equal(position_ids, expected)


def test_encoder_version_is_bumped_for_position_ids() -> None:
    assert ENCODER_VERSION >= 2
