#!/usr/bin/env python3
"""Check GPU VRAM usage for a prefill encoder (issue #11)."""

from __future__ import annotations

import argparse
import gc
import logging

import torch

from genrec_lite.config import find_project_root, load_llm_config
from genrec_lite.encode.prefill import PrefillEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_once(
    model: str,
    batch_size: int,
    seq_len: int,
    dtype: str | None,
    quantize: str | None,
) -> float:
    root = find_project_root()
    llm_cfg = load_llm_config(model, config_dir=root / "configs")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    encoder = PrefillEncoder.from_config(llm_cfg, device="cuda")
    texts = ["vram check prompt " * max(1, seq_len // 16)] * batch_size
    encoder.encode_batch(texts)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    logger.info(
        "batch=%d seq=%d peak=%.2f GB reserved=%.2f GB",
        batch_size,
        seq_len,
        peak_gb,
        reserved_gb,
    )
    return peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VRAM usage for prefill encoding")
    parser.add_argument("--model", required=True, help="LLM config name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--find-max-batch", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available on this machine")
        return 1

    if not args.find_max_batch:
        _run_once(args.model, args.batch_size, args.seq_len, args.dtype, args.quantize)
        return 0

    low, high = 1, 64
    best = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            _run_once(args.model, mid, args.seq_len, args.dtype, args.quantize)
            best = mid
            low = mid + 1
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                high = mid - 1
            else:
                raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Max batch size without OOM: %d", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
