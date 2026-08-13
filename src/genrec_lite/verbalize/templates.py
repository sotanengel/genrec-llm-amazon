"""Jinja2 verbalizer templates (DESIGN.md §5)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import polars as pl
from jinja2 import Template

from genrec_lite.config import VerbalizerYamlConfig
from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.budget import count_tokens, truncate_to_budget
from genrec_lite.verbalize.compress import CompressConfig, compress_history_events

PREFIX_TEMPLATE = Template(
    """You are a product recommendation ranker.

## User profile
Interactions: {{ n_inter }} | Active since: {{ first_seen_relative }} | \
Repeat ratio: {{ repeat_ratio }}
Top categories: {{ top3_categories }}

## Purchase history (most recent first)
"""
)

FULL_EVENT_TEMPLATE = Template(
    """{{ idx }}. [{{ days_ago }}d ago] {{ title }} | {{ category_leaf }} | \
${{ price }} | rated {{ rating }}/5
{% if description %}{{ description }}
{% endif %}"""
)

COMPACT_EVENT_TEMPLATE = Template(
    """{{ idx }}. [{{ days_ago }}d ago] {{ title }} | {{ category_leaf }}"""
)

IDS_EVENT_TEMPLATE = Template("""{{ idx }}. item_{{ item_id }}""")

FOOTER_TEMPLATE = Template(
    """
## Current context
Time: {{ weekday }} {{ hour }}:00 | Season: {{ season }}

## Task
Predict the next product this user will purchase."""
)

NO_CONTEXT_FOOTER = Template(
    """
## Task
Predict the next product this user will purchase."""
)


@dataclass
class VerbalizerConfig:
    name: str
    variant: str = "v1_full"
    compress: CompressConfig | None = None
    include_context: bool = True
    include_descriptions: bool = True


def _category_leaf(category_path: str) -> str:
    if not category_path:
        return "Unknown"
    parts = [p.strip() for p in category_path.split(">")]
    return parts[-1] if parts else "Unknown"


def _season_from_ts(ts: int) -> str:
    month = dt.datetime.fromtimestamp(ts, tz=dt.UTC).month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _build_item_lookup(items: pl.DataFrame) -> dict[int, dict[str, Any]]:
    return {int(r["item_id"]): r for r in items.iter_rows(named=True)}


def _build_user_lookup(users: pl.DataFrame) -> dict[int, dict[str, Any]]:
    return {int(r["user_id"]): r for r in users.iter_rows(named=True)}


def _build_history_events(
    sample: Sample,
    item_lookup: dict[int, dict[str, Any]],
    config: VerbalizerConfig,
) -> list[dict[str, Any]]:
    history = list(reversed(sample.history))
    events: list[dict[str, Any]] = []
    for item_id in history:
        item = item_lookup.get(int(item_id), {})
        events.append(
            {
                "item_id": int(item_id),
                "title": str(item.get("title", f"item_{item_id}")),
                "category_leaf": _category_leaf(str(item.get("category_path", ""))),
                "price": item.get("price"),
                "description": str(item.get("description", ""))[:120],
                "rating": None,
                "days_ago": max(0, (sample.cutoff_ts - int(item.get("first_seen_ts", 0))) // 86400),
            }
        )
    compress_cfg = config.compress or CompressConfig()
    if not config.include_descriptions:
        compress_cfg.desc_top_k = 0
    return compress_history_events(events, compress_cfg)


def _top_categories(history: list[int], item_lookup: dict[int, dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item_id in history:
        item = item_lookup.get(int(item_id), {})
        leaf = _category_leaf(str(item.get("category_path", "")))
        counts[leaf] = counts.get(leaf, 0) + 1
    top3 = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:3]
    return ", ".join(name for name, _ in top3) if top3 else "Unknown"


class TemplateVerbalizer:
    """Configurable verbalizer using Jinja2 templates."""

    def __init__(self, config: VerbalizerConfig) -> None:
        self._config = config
        self._items_id: int | None = None
        self._users_id: int | None = None
        self._item_lookup: dict[int, dict[str, Any]] = {}
        self._user_lookup: dict[int, dict[str, Any]] = {}

    def name(self) -> str:
        return self._config.name

    def _ensure_lookups(self, items: pl.DataFrame, users: pl.DataFrame) -> None:
        items_id = id(items)
        if self._items_id != items_id:
            self._items_id = items_id
            self._item_lookup = _build_item_lookup(items)
        users_id = id(users)
        if self._users_id != users_id:
            self._users_id = users_id
            self._user_lookup = _build_user_lookup(users)

    def render(
        self,
        sample: Sample,
        items: pl.DataFrame,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
        budget: TokenBudget,
    ) -> str:
        self._ensure_lookups(items, users)
        user = self._user_lookup.get(sample.user_id, {})
        first_ts = int(user.get("first_ts", sample.cutoff_ts))
        days_active = max(0, (sample.cutoff_ts - first_ts) // 86400)
        prefix = PREFIX_TEMPLATE.render(
            n_inter=int(user.get("n_inter", len(sample.history))),
            first_seen_relative=f"{days_active}d ago",
            repeat_ratio=f"{float(user.get('repeat_ratio', 0.0)):.2f}",
            top3_categories=_top_categories(sample.history, self._item_lookup),
        )

        events = _build_history_events(sample, self._item_lookup, self._config)
        history_lines: list[str] = []
        for idx, event in enumerate(events, start=1):
            if self._config.variant == "v0_ids_only":
                history_lines.append(IDS_EVENT_TEMPLATE.render(idx=idx, item_id=event["item_id"]))
            elif self._config.variant == "v2_compact":
                history_lines.append(COMPACT_EVENT_TEMPLATE.render(idx=idx, **event))
            else:
                history_lines.append(FULL_EVENT_TEMPLATE.render(idx=idx, **event))

        cutoff_dt = dt.datetime.fromtimestamp(sample.cutoff_ts, tz=dt.UTC)
        if self._config.include_context:
            footer = FOOTER_TEMPLATE.render(
                weekday=cutoff_dt.strftime("%A"),
                hour=cutoff_dt.hour,
                season=_season_from_ts(sample.cutoff_ts),
            )
        else:
            footer = NO_CONTEXT_FOOTER.render()

        prompt = prefix + "\n".join(history_lines) + footer
        if count_tokens(prompt, budget.tokenizer_name) > budget.max_tokens:
            tighter = VerbalizerConfig(
                name=self._config.name,
                variant=self._config.variant,
                compress=CompressConfig(
                    max_history=max(
                        3, (self._config.compress or CompressConfig()).max_history // 2
                    ),
                    desc_top_k=0,
                    title_max_chars=40,
                ),
                include_context=self._config.include_context,
                include_descriptions=False,
            )
            tighter_verbalizer = TemplateVerbalizer(tighter)
            tighter_verbalizer._items_id = self._items_id
            tighter_verbalizer._users_id = self._users_id
            tighter_verbalizer._item_lookup = self._item_lookup
            tighter_verbalizer._user_lookup = self._user_lookup
            return tighter_verbalizer.render(sample, items, users, interactions, budget)
        return truncate_to_budget(prompt, budget.max_tokens, budget.tokenizer_name)


def build_verbalizer_from_config(cfg: VerbalizerYamlConfig) -> TemplateVerbalizer:
    """Build a verbalizer that honors YAML settings (max_history, desc_top_k, etc.)."""
    compress = CompressConfig(
        max_history=cfg.max_history,
        desc_top_k=cfg.desc_top_k,
        title_max_chars=cfg.title_max_chars,
    )
    return TemplateVerbalizer(
        VerbalizerConfig(
            name=cfg.name,
            variant=cfg.variant,
            compress=compress,
            include_context=cfg.include_context,
            include_descriptions=cfg.include_descriptions,
        )
    )


def build_verbalizer(name: str) -> TemplateVerbalizer:
    """Build a verbalizer from a preset config name."""
    variants = {
        "v0_ids_only": VerbalizerConfig(
            name="v0_ids_only", variant="v0_ids_only", include_context=True
        ),
        "v1_full": VerbalizerConfig(name="v1_full", variant="v1_full"),
        "v2_compact": VerbalizerConfig(
            name="v2_compact", variant="v2_compact", include_descriptions=False
        ),
        "v3_no_context": VerbalizerConfig(
            name="v3_no_context", variant="v1_full", include_context=False
        ),
    }
    if name not in variants:
        raise ValueError(f"Unknown verbalizer: {name}")
    return TemplateVerbalizer(variants[name])
