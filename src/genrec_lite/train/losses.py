"""Training losses (DESIGN.md §7.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def sampled_softmax_with_logq(
    scores: Tensor,
    target_idx: Tensor,
    log_q: Tensor | None = None,
) -> Tensor:
    """Catalog-constrained CE on a candidate score matrix with optional logQ correction."""
    adjusted = scores if log_q is None else scores - log_q
    return torch.nn.functional.cross_entropy(adjusted, target_idx)


def reward_weighted_loss(per_sample_loss: Tensor, rewards: Tensor) -> Tensor:
    """Sum of reward-weighted per-sample losses (DESIGN.md §7.1 L_reward)."""
    return (per_sample_loss * rewards).sum()
