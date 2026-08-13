#!/usr/bin/env python3
"""Benchmark prefill throughput (DESIGN.md §2.3, M2)."""

from __future__ import annotations

import argparse
import logging
import time

from genrec_lite.encode.prefill import PrefillEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLM prefill throughput")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    encoder = PrefillEncoder(model_id=args.model_id, dtype="float32", max_len=args.seq_len)
    text = "benchmark " * (args.seq_len // 2)
    texts = [text] * args.batch_size

    for _ in range(args.warmup):
        encoder.encode_batch(texts)

    start = time.perf_counter()
    total_tokens = 0
    for _ in range(args.steps):
        encoder.encode_batch(texts)
        total_tokens += args.batch_size * args.seq_len
    elapsed = time.perf_counter() - start
    tok_s = total_tokens / elapsed if elapsed > 0 else 0.0
    logger.info(
        "model=%s batch=%d seq=%d -> %.1f tok/s (%.3fs total)",
        args.model_id,
        args.batch_size,
        args.seq_len,
        tok_s,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
