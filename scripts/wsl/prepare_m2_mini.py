"""Prepare a synthetic mini Amazon bundle for M2 GPU acceptance checks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from genrec_lite.data.loaders.amazon import prepare_from_records


def main() -> None:
    random.seed(42)
    n_users, n_items = 20, 40
    base_ts = 1_600_000_000_000
    reviews: list[dict[str, Any]] = []
    # Ensure each user has >= min_core interactions with distinct timestamps.
    idx = 0
    for user in range(n_users):
        for j in range(8):
            reviews.append(
                {
                    "user_id": f"user_{user}",
                    "parent_asin": f"item_{(user + j) % n_items}",
                    "timestamp": base_ts + idx * 86_400_000,
                    "rating": float(3 + (j % 3)),
                    "main_category": "Video_Games",
                }
            )
            idx += 1
    meta: list[dict[str, Any]] = [
        {
            "parent_asin": f"item_{i}",
            "title": f"Game Title {i}",
            "store": f"Brand{i}",
            "main_category": "Video_Games",
            "categories": [["Electronics", "Video Games", f"Subcat{i}"]],
            "price": 29.99 + i,
            "description": f"Description for item {i} " * 10,
        }
        for i in range(n_items)
    ]
    out = Path("data/processed/amazon_video_games")
    prepare_from_records(
        reviews,
        meta,
        out,
        split_strategy="global_temporal",
        cold_threshold=5,
        min_core=3,
    )
    print(f"Prepared {out}")


if __name__ == "__main__":
    main()
