"""Dataset statistics (DESIGN.md M0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

from genrec_lite.data.schema import read_parquet_bundle


@dataclass(frozen=True)
class DatasetStats:
    n_users: int
    n_items: int
    n_interactions: int
    density: float
    avg_repeat_ratio: float
    ts_min: int
    ts_max: int
    n_train: int
    n_valid: int
    n_test: int


def compute_stats(data_dir: Path) -> DatasetStats:
    interactions, items, users, _ = read_parquet_bundle(data_dir)
    n_users = users.height
    n_items = items.height
    n_inter = interactions.height
    density = n_inter / (n_users * n_items) if n_users * n_items > 0 else 0.0
    repeat_mean = users["repeat_ratio"].mean()
    avg_repeat = float(cast(float, repeat_mean)) if repeat_mean is not None else 0.0

    split_counts = interactions.group_by("split").len()
    count_map = {row["split"]: row["len"] for row in split_counts.iter_rows(named=True)}

    ts_min_val = interactions["ts"].min()
    ts_max_val = interactions["ts"].max()
    if ts_min_val is None or ts_max_val is None:
        raise ValueError("interactions table has no timestamps")

    return DatasetStats(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_inter,
        density=density,
        avg_repeat_ratio=avg_repeat,
        ts_min=cast(int, ts_min_val),
        ts_max=cast(int, ts_max_val),
        n_train=count_map.get(0, 0),
        n_valid=count_map.get(1, 0),
        n_test=count_map.get(2, 0),
    )


def print_stats(data_dir: Path, console: Console | None = None) -> DatasetStats:
    stats = compute_stats(data_dir)
    out = console or Console()
    table = Table(title=f"Dataset stats: {data_dir}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Users", str(stats.n_users))
    table.add_row("Items", str(stats.n_items))
    table.add_row("Interactions", str(stats.n_interactions))
    table.add_row("Density", f"{stats.density:.6f}")
    table.add_row("Avg repeat ratio", f"{stats.avg_repeat_ratio:.4f}")
    table.add_row("Time range", f"{stats.ts_min} - {stats.ts_max}")
    table.add_row("Train / Valid / Test", f"{stats.n_train} / {stats.n_valid} / {stats.n_test}")
    out.print(table)
    return stats


def stats_to_dict(stats: DatasetStats) -> dict[str, float | int]:
    return {
        "n_users": stats.n_users,
        "n_items": stats.n_items,
        "n_interactions": stats.n_interactions,
        "density": stats.density,
        "avg_repeat_ratio": stats.avg_repeat_ratio,
        "ts_min": stats.ts_min,
        "ts_max": stats.ts_max,
        "n_train": stats.n_train,
        "n_valid": stats.n_valid,
        "n_test": stats.n_test,
    }
