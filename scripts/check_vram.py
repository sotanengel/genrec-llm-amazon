#!/usr/bin/env python3
"""Check GPU VRAM usage for a prefill encoder."""

from __future__ import annotations

import argparse
import logging

import torch

from genrec_lite.encode.prefill import PrefillEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VRAM usage for prefill encoding")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available on this machine")
        return 1

    torch.cuda.reset_peak_memory_stats()
    encoder = PrefillEncoder(
        model_id=args.model_id,
        dtype=args.dtype,
        max_len=args.seq_len,
        device="cuda",
    )
    texts = ["vram check prompt"] * args.batch_size
    encoder.encode_batch(texts)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info("Peak VRAM: %.2f GB", peak_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
