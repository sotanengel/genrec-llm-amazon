#!/usr/bin/env python3
"""Profile verbalizer CPU stage and token-length histogram (issue #11)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from genrec_lite.config import find_project_root, load_data_config, load_verbalizer_config
from genrec_lite.data.schema import read_parquet_bundle
from genrec_lite.verbalize.base import TokenBudget
from genrec_lite.verbalize.budget import count_tokens
from genrec_lite.verbalize.templates import build_verbalizer_from_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile verbalizer CPU cost")
    parser.add_argument("--dataset", default="amazon_video_games")
    parser.add_argument("--verbalizer", default="v1_full")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    root = find_project_root()
    data_config = load_data_config(args.dataset, config_dir=root / "configs")
    verb_config = load_verbalizer_config(args.verbalizer, config_dir=root / "configs")
    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return 1

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    renderer = build_verbalizer_from_config(verb_config)
    tokenizer_name = verb_config.tokenizer_name or "gpt2"
    budget = TokenBudget(max_tokens=verb_config.max_tokens, tokenizer_name=tokenizer_name)

    from genrec_lite.cli import _sample_from_row

    subset = samples.head(args.n)
    lengths: list[int] = []
    start = time.perf_counter()
    for row in subset.iter_rows(named=True):
        sample = _sample_from_row(row)
        prompt = renderer.render(sample, items, users, interactions, budget)
        lengths.append(count_tokens(prompt, tokenizer_name))
    elapsed = time.perf_counter() - start

    hist = Counter((length // 32) * 32 for length in lengths)
    payload = {
        "dataset": args.dataset,
        "verbalizer": args.verbalizer,
        "n_samples": len(lengths),
        "elapsed_s": elapsed,
        "samples_per_s": len(lengths) / elapsed if elapsed > 0 else 0.0,
        "token_length_histogram": dict(sorted(hist.items())),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    out_path = (
        Path(args.json_path)
        if args.json_path
        else root / "reports/bench" / f"verbalize_{args.verbalizer}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
