"""GenRecLite model tests (M3, DESIGN.md §14.2)."""

from __future__ import annotations

import torch

from genrec_lite.models.genrec_lite import GenRecLite


def test_score_shape_full_catalog() -> None:
    batch, d_llm, d_emb, n_items = 4, 32, 16, 50
    model = GenRecLite(d_llm=d_llm, d_emb=d_emb, n_items=n_items)
    h = torch.randn(batch, d_llm)
    scores = model.score(h)
    assert scores.shape == (batch, n_items)


def test_score_shape_candidate_set() -> None:
    batch, d_llm, d_emb, n_items, k = 3, 32, 16, 50, 7
    model = GenRecLite(d_llm=d_llm, d_emb=d_emb, n_items=n_items)
    h = torch.randn(batch, d_llm)
    candidate_ids = torch.randint(0, n_items, (batch, k))
    scores = model.score(h, candidate_ids=candidate_ids)
    assert scores.shape == (batch, k)


def test_dot_equivalence() -> None:
    batch, d_llm, d_emb, n_items = 2, 8, 4, 10
    torch.manual_seed(0)
    model = GenRecLite(d_llm=d_llm, d_emb=d_emb, n_items=n_items, scorer="dot")
    model.eval()
    h = torch.randn(batch, d_llm)
    with torch.no_grad():
        z = model.head.dropout(model.head.proj(h))
        expected = z @ model.head.item_emb.weight.T
    actual = model.score(h)
    assert torch.allclose(actual, expected)


def test_text_init_uses_provided_matrix() -> None:
    n_items, d_emb = 12, 8
    item_init = torch.arange(n_items * d_emb, dtype=torch.float32).reshape(n_items, d_emb)
    model = GenRecLite(d_llm=16, d_emb=d_emb, n_items=n_items, item_init=item_init)
    assert torch.equal(model.head.item_emb.weight.data, item_init)


def test_text_init_shape_projection() -> None:
    n_items, d_init, d_emb = 6, 10, 4
    torch.manual_seed(1)
    item_init = torch.randn(n_items, d_init)
    model = GenRecLite(d_llm=16, d_emb=d_emb, n_items=n_items, item_init=item_init)
    with torch.no_grad():
        expected = model.head.item_init_proj(item_init)
    assert model.head.item_emb.weight.shape == (n_items, d_emb)
    assert torch.allclose(model.head.item_emb.weight.data, expected)


def test_backward_updates_head_only_when_llm_frozen() -> None:
    batch, d_llm, d_emb, n_items = 2, 8, 4, 10
    model = GenRecLite(d_llm=d_llm, d_emb=d_emb, n_items=n_items)
    h = torch.randn(batch, d_llm, requires_grad=False)
    scores = model.score(h)
    loss = scores.sum()
    loss.backward()
    assert h.grad is None
    assert model.head.proj.weight.grad is not None
    assert model.head.item_emb.weight.grad is not None
