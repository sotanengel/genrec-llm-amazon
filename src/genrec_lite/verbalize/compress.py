"""Context compression rules (DESIGN.md §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompressConfig:
    max_history: int = 20
    min_rating: float = 0.0
    min_price: float = 0.0
    collapse_repeats: bool = True
    desc_top_k: int = 3
    title_max_chars: int = 60


def _truncate_title(title: str, max_chars: int) -> str:
    if len(title) <= max_chars:
        return title
    return title[: max_chars - 3] + "..."


def compress_history_events(
    events: list[dict[str, Any]],
    config: CompressConfig,
) -> list[dict[str, Any]]:
    """Apply priority-ordered compression to history events (most recent first)."""
    filtered = events

    # Priority 2: drop weak signals
    if config.min_rating > 0 or config.min_price > 0:
        filtered = [
            e
            for e in filtered
            if (e.get("rating") is None or float(e.get("rating", 0)) >= config.min_rating)
            and (e.get("price") is None or float(e.get("price", 0)) >= config.min_price)
        ]

    # Priority 1: keep only recent N
    filtered = filtered[: config.max_history]

    # Priority 3: collapse repeats
    if config.collapse_repeats:
        collapsed: list[dict[str, Any]] = []
        for event in filtered:
            if collapsed and collapsed[-1]["item_id"] == event["item_id"]:
                collapsed[-1]["repeat_count"] = collapsed[-1].get("repeat_count", 1) + 1
            else:
                collapsed.append(dict(event))
        filtered = collapsed

    # Priority 4: drop description except top-k recent
    for i, event in enumerate(filtered):
        if i >= config.desc_top_k:
            event["description"] = ""

    # Priority 5: truncate titles
    for event in filtered:
        if "title" in event:
            event["title"] = _truncate_title(str(event["title"]), config.title_max_chars)

    return filtered
