"""Verbalizer CPU-hotpath performance regression tests (issue #10 / U3, DESIGN.md §14.2).

These tests assert *call counts*, not wall-clock time, so they stay stable in CI:
`_build_history_events`/`_top_categories`/`render` must not rebuild the full
`items` lookup or re-scan `users` for every sample. They must FAIL against the
pre-fix implementation (which calls `items.iter_rows` and `users.filter` once
per sample) and PASS once the lookups are hoisted and reused.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl
import pytest

from genrec_lite.config import VerbalizerYamlConfig
from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.templates import build_verbalizer, build_verbalizer_from_config

N_ITEMS = 5_000
N_SAMPLES = 200


def _synthetic_catalog(n_items: int = N_ITEMS) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    items = pl.DataFrame(
        {
            "item_id": list(range(n_items)),
            "title": [f"Item Title {i}" for i in range(n_items)],
            "brand": [f"Brand{i % 50}" for i in range(n_items)],
            "category_path": [f"Electronics > Games > Cat{i % 20}" for i in range(n_items)],
            "price": [float(10 + (i % 100)) for i in range(n_items)],
            "description": [f"Description for item {i} " * 5 for i in range(n_items)],
            "first_seen_ts": [1_600_000_000 + i * 1_000 for i in range(n_items)],
            "n_train_inter": [10] * n_items,
        }
    )
    users = pl.DataFrame(
        {
            "user_id": [0],
            "raw_id": ["user_0"],
            "n_inter": [5],
            "first_ts": [1_600_000_000],
            "last_ts": [1_600_900_000],
            "repeat_ratio": [0.1],
        }
    )
    interactions = pl.DataFrame(
        {"user_id": [0], "item_id": [0], "ts": [1_600_000_000], "split": [0]}
    )
    return items, users, interactions


def _synthetic_samples(n: int = N_SAMPLES, n_items: int = N_ITEMS) -> list[Sample]:
    samples: list[Sample] = []
    for i in range(n):
        history = [(i + j) % n_items for j in range(5)]
        samples.append(
            Sample(
                user_id=0,
                cutoff_ts=1_600_950_000,
                target_item=(i + 5) % n_items,
                history=history,
                is_repeat=False,
                target_is_cold=False,
            )
        )
    return samples


def test_render_hoists_item_and_user_lookups_across_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering many samples against one catalog must not rescan it per sample."""
    items, users, interactions = _synthetic_catalog()
    samples = _synthetic_samples()
    # Deliberately huge budget: never trigger the overflow/re-render path, so
    # this test isolates the base per-sample lookup cost from compression retries.
    budget = TokenBudget(max_tokens=100_000, tokenizer_name="gpt2")
    verbalizer = build_verbalizer("v1_full")

    call_counts: Counter[str] = Counter()

    original_iter_rows = pl.DataFrame.iter_rows

    def counting_iter_rows(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if self is items:
            call_counts["items_iter_rows"] += 1
        elif self is users:
            call_counts["users_iter_rows"] += 1
        return original_iter_rows(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "iter_rows", counting_iter_rows)

    original_filter = pl.DataFrame.filter

    def counting_filter(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if self is users:
            call_counts["users_filter"] += 1
        return original_filter(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "filter", counting_filter)

    for sample in samples:
        verbalizer.render(sample, items, users, interactions, budget)

    assert call_counts["items_iter_rows"] <= 1, (
        f"items.iter_rows() was called {call_counts['items_iter_rows']} times "
        f"across {N_SAMPLES} renders; expected the item lookup to be built once "
        "and reused, not rebuilt per sample."
    )
    assert call_counts["users_filter"] == 0, (
        f"users.filter() was called {call_counts['users_filter']} times across "
        f"{N_SAMPLES} renders; expected a hoisted lookup instead of a per-sample "
        "DataFrame scan."
    )


def test_build_verbalizer_from_config_respects_max_history() -> None:
    items, users, interactions = _synthetic_catalog(n_items=50)
    sample = Sample(
        user_id=0,
        cutoff_ts=1_600_950_000,
        target_item=10,
        history=list(range(20)),
        is_repeat=False,
        target_is_cold=False,
    )
    cfg = VerbalizerYamlConfig(
        name="v1_full",
        variant="v1_full",
        max_history=3,
        max_tokens=100_000,
    )
    verbalizer = build_verbalizer_from_config(cfg)
    budget = TokenBudget(max_tokens=100_000, tokenizer_name="gpt2")
    prompt = verbalizer.render(sample, items, users, interactions, budget)
    history_lines = [
        line for line in prompt.splitlines() if line and line[0].isdigit() and ". " in line[:4]
    ]
    assert len(history_lines) <= 3


def test_lookup_hoist_preserves_prompt_bytes() -> None:
    """O(n×m) lookup hoisting must not change rendered prompt text."""
    items, users, interactions = _synthetic_catalog(n_items=20)
    sample = _synthetic_samples(n=1, n_items=20)[0]
    budget = TokenBudget(max_tokens=100_000, tokenizer_name="gpt2")
    preset = build_verbalizer("v1_full")
    golden = preset.render(sample, items, users, interactions, budget)
    cfg = VerbalizerYamlConfig(name="v1_full", variant="v1_full", max_tokens=100_000)
    from_config = build_verbalizer_from_config(cfg)
    assert from_config.render(sample, items, users, interactions, budget) == golden
