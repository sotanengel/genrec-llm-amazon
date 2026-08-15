"""Loss function tests (M3, DESIGN.md §14.2)."""

from __future__ import annotations

import torch

from genrec_lite.train.losses import reward_weighted_loss, sampled_softmax_with_logq


def test_sampled_softmax_matches_full_when_all_sampled() -> None:
    batch, n_items = 3, 5
    torch.manual_seed(0)
    scores = torch.randn(batch, n_items, requires_grad=True)
    target_idx = torch.tensor([1, 3, 0])
    sampled = sampled_softmax_with_logq(scores, target_idx)
    expected = torch.nn.functional.cross_entropy(scores, target_idx)
    assert torch.allclose(sampled, expected)


def test_logq_correction_unbiased_gradient() -> None:
    batch, k = 2, 4
    torch.manual_seed(1)
    scores = torch.randn(batch, k, requires_grad=True)
    target_idx = torch.tensor([0, 2])
    uniform_log_q = torch.full((batch, k), -torch.log(torch.tensor(float(k))))
    loss_plain = sampled_softmax_with_logq(scores, target_idx)
    loss_corrected = sampled_softmax_with_logq(scores, target_idx, log_q=uniform_log_q)
    loss_plain.backward(retain_graph=True)
    grad_plain = scores.grad.clone()
    scores.grad = None
    loss_corrected.backward()
    assert torch.allclose(grad_plain, scores.grad)


def test_reward_weight_scales_loss() -> None:
    per_sample = torch.tensor([1.0, 2.0, 3.0])
    rewards = torch.tensor([1.0, 2.0, 1.0])
    base = reward_weighted_loss(per_sample, rewards)
    scaled = reward_weighted_loss(per_sample, rewards * 2)
    assert torch.allclose(scaled, base * 2)


def test_reward_zero_masks_sample() -> None:
    batch, k = 3, 3
    torch.manual_seed(2)
    scores = torch.randn(batch, k, requires_grad=True)
    target_idx = torch.tensor([0, 1, 2])
    per_sample = torch.stack(
        [sampled_softmax_with_logq(scores[i : i + 1], target_idx[i : i + 1]) for i in range(batch)]
    )
    rewards = torch.tensor([1.0, 0.0, 1.0])
    loss = reward_weighted_loss(per_sample, rewards)
    loss.backward()
    assert scores.grad is not None
    assert torch.allclose(scores.grad[1], torch.zeros_like(scores.grad[1]))
