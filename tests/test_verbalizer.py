"""Verbalizer tests (DESIGN.md §14.2)."""

from __future__ import annotations

import polars as pl

from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.budget import count_tokens, get_tokenizer
from genrec_lite.verbalize.compress import CompressConfig, compress_history_events
from genrec_lite.verbalize.templates import build_verbalizer


def _mini_sample() -> Sample:
    return Sample(
        user_id=0,
        cutoff_ts=1_600_100_000,
        target_item=1,
        history=[0, 1, 2],
        is_repeat=False,
        target_is_cold=False,
    )


def _mini_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    items = pl.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["Game A", "Game B", "Game C"],
            "brand": ["BrandA", "BrandB", "BrandC"],
            "category_path": ["Electronics > Games > A"] * 3,
            "price": [10.0, 20.0, 30.0],
            "description": ["Desc A", "Desc B", "Desc C"],
            "first_seen_ts": [1_600_000_000, 1_600_010_000, 1_600_020_000],
            "n_train_inter": [10, 10, 10],
        }
    )
    users = pl.DataFrame(
        {
            "user_id": [0],
            "raw_id": ["user_0"],
            "n_inter": [3],
            "first_ts": [1_600_000_000],
            "last_ts": [1_600_100_000],
            "repeat_ratio": [0.1],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": [0, 0, 0],
            "item_id": [0, 1, 2],
            "ts": [1_600_000_000, 1_600_050_000, 1_600_090_000],
            "split": [0, 0, 0],
        }
    )
    return items, users, interactions


def test_render_deterministic(tiny_model_id: str) -> None:
    sample = _mini_sample()
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v1_full")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    out1 = verbalizer.render(sample, items, users, interactions, budget)
    out2 = verbalizer.render(sample, items, users, interactions, budget)
    assert out1 == out2


def test_compression_reduces_below_budget(tiny_model_id: str) -> None:
    events = [
        {
            "item_id": i,
            "title": f"Very Long Game Title Number {i}" * 3,
            "category_leaf": "Games",
            "price": 10.0,
            "description": "Long description " * 20,
            "rating": 3.0,
        }
        for i in range(30)
    ]
    compressed = compress_history_events(events, CompressConfig(max_history=5, desc_top_k=0))
    assert len(compressed) <= 5
    text = "\n".join(str(e["title"]) for e in compressed)
    assert count_tokens(text, tiny_model_id) <= 256


def test_compression_priority_order() -> None:
    events = [
        {"item_id": 1, "title": "A", "rating": 1.0, "price": 1.0, "description": "d"},
        {"item_id": 2, "title": "B", "rating": 5.0, "price": 50.0, "description": "d"},
    ]
    cfg = CompressConfig(max_history=10, min_rating=4.0, min_price=10.0)
    compressed = compress_history_events(events, cfg)
    assert len(compressed) == 1
    assert compressed[0]["item_id"] == 2


def test_id_only_verbalizer_contains_no_titles(tiny_model_id: str) -> None:
    sample = _mini_sample()
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v0_ids_only")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    prompt = verbalizer.render(sample, items, users, interactions, budget)
    assert "Game A" not in prompt
    assert "item_0" in prompt


def test_prompt_prefix_is_shared(tiny_model_id: str) -> None:
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v1_full")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    history = [0, 1, 2]
    cutoff = 1_600_100_000
    sample_a = Sample(0, cutoff, 1, history, False, False)
    sample_b = Sample(0, cutoff, 2, history, True, False)
    prompt_a = verbalizer.render(sample_a, items, users, interactions, budget)
    prompt_b = verbalizer.render(sample_b, items, users, interactions, budget)
    tokenizer = get_tokenizer(tiny_model_id)
    prefix_a = tokenizer.encode(prompt_a, add_special_tokens=False)[:100]
    prefix_b = tokenizer.encode(prompt_b, add_special_tokens=False)[:100]
    assert prefix_a == prefix_b
