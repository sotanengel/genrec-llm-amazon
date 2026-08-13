#!/usr/bin/env python3
"""Benchmark prefill throughput (DESIGN.md §2.3, issue #11)."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from genrec_lite.config import find_project_root, load_llm_config
from genrec_lite.encode.prefill import PrefillEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _memory_stats(device: str) -> dict[str, float]:
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        "free_gb": free / (1024**3),
        "total_gb": total / (1024**3),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = find_project_root()
    if args.dry_run_cpu:
        model_id = "sshleifer/tiny-gpt2"
        dtype = "float32"
        device = "cpu"
        revision = None
        max_len = args.seq_len
        quantize = None
    else:
        llm_cfg = load_llm_config(args.model, config_dir=root / "configs")
        model_id = llm_cfg.model_id
        dtype = args.dtype or llm_cfg.dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        revision = llm_cfg.revision
        max_len = args.seq_len or llm_cfg.max_len
        quantize = args.quantize or llm_cfg.quantize

    encoder = PrefillEncoder(
        model_id=model_id,
        dtype=dtype,
        max_len=max_len,
        quantize=quantize,
        device=device,
        revision=revision,
    )
    text = "benchmark " * max(1, max_len // 8)
    texts = [text] * args.batch_size
    actual_lengths = encoder.token_lengths(texts)
    real_tokens = sum(actual_lengths)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.warmup):
        encoder.encode_batch(texts, padding=args.padding)
        if device == "cuda":
            torch.cuda.synchronize()

    step_ms: list[float] = []
    for _ in range(args.steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        encoder.encode_batch(texts, padding=args.padding)
        if device == "cuda":
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - start) * 1000.0)

    padded_tokens = args.batch_size * max_len * args.steps
    real_total_tokens = real_tokens * args.steps
    elapsed_s = sum(step_ms) / 1000.0
    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "batch_size": args.batch_size,
        "seq_len": max_len,
        "padding": args.padding,
        "dtype": dtype,
        "quantize": quantize,
        "steps": args.steps,
        "warmup": args.warmup,
        "tok_per_s_padded": padded_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "tok_per_s_real": real_total_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "ms_per_batch_p50": statistics.median(step_ms),
        "ms_per_batch_p90": _percentile(step_ms, 90.0),
        "memory": _memory_stats(device),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLM prefill throughput")
    parser.add_argument("--model", help="LLM config name under configs/model/llm/")
    parser.add_argument("--dry-run-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if not args.dry_run_cpu and not args.model:
        parser.error("Either --model or --dry-run-cpu is required")

    if args.sweep and not args.dry_run_cpu:
        batch_sizes = [1, 2, 4, 8]
        results = [
            run_benchmark(argparse.Namespace(**{**vars(args), "batch_size": bs}))
            for bs in batch_sizes
        ]
        payload = {"sweep": results}
    else:
        payload = run_benchmark(args)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
