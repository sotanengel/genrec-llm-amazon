#!/usr/bin/env python3
"""Benchmark prefill throughput (DESIGN.md §2.3, issue #11).

Correctness note (measurement bug fix, 2026-08-14): earlier versions of this
script generated filler text using a fixed "8 characters per token" guess
(``"benchmark " * (max_len // 8)``), which for real tokenizers produced
sequences far shorter than the requested ``--seq-len`` (e.g. ~64-128 tokens
for a nominal 512). Since attention cost and peak VRAM scale with the *real*
sequence length, every recorded benchmark understated the true cost of the
seq-len it claimed to measure. Text generation below instead grows a varied,
deterministic filler string and re-tokenizes it (no fixed ratio assumed)
until it verifiably reaches ``max_len`` tokens. Similarly, ``tok_per_s_padded``
used to be computed from the nominal ``max_len`` rather than the actual
padded tensor width produced by the tokenizer under the batch's padding mode,
inflating throughput by however much the batch was shorter than ``max_len``.
It is now computed from the real tensor width.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
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

# Minimum fraction of the requested --seq-len that the achieved (real,
# truncated) token count must reach before we consider the benchmark to
# actually be measuring what it claims. Below this, a WARNING is logged and
# recorded in the JSON payload -- a silently-short benchmark is exactly what
# caused the original measurement bug.
SEQ_LEN_WARN_THRESHOLD = 0.95

# Varied vocabulary for filler text. A single repeated token (e.g. the old
# "benchmark benchmark benchmark ...") is an unrepresentative attention
# workload, so filler is built from a mix of common words instead.
_FILLER_VOCAB = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "market",
    "analysis",
    "reveals",
    "significant",
    "trends",
    "customer",
    "reviews",
    "product",
    "quality",
    "shipping",
    "delivery",
    "experience",
    "recommend",
    "purchase",
    "value",
    "price",
    "comparison",
    "feature",
    "design",
    "material",
    "durability",
    "performance",
    "warranty",
    "support",
    "documentation",
    "installation",
    "compatibility",
    "software",
    "hardware",
    "update",
    "version",
    "release",
    "notes",
    "improvement",
    "interface",
    "workflow",
    "efficiency",
    "algorithm",
    "model",
    "training",
    "evaluation",
    "benchmark",
    "throughput",
    "latency",
    "memory",
]


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


def build_filler_text(encoder: PrefillEncoder, target_tokens: int, seed: int = 0) -> str:
    """Build varied, deterministic filler text that tokenizes to >= target_tokens.

    No fixed tokens-per-word ratio is assumed (it differs per tokenizer, e.g.
    GPT-2 BPE vs Qwen3's), so this grows the text and re-tokenizes (without
    truncation) until the real token count reaches the target. Truncating the
    result at ``max_length=target_tokens`` (as ``encode_batch``/``token_lengths``
    do) then yields exactly ``target_tokens`` real, non-pad tokens.
    """
    if target_tokens <= 0:
        return "benchmark"

    rng = random.Random(seed)
    words: list[str] = []
    n_tokens = 0

    # Coarse phase: add words in batches to avoid one tokenizer call per word
    # for large targets. Each batch is sized conservatively (target // 8, so
    # even a >=8-tokens-per-word tokenizer won't overshoot wildly before the
    # fine phase takes over).
    coarse_batch = max(1, target_tokens // 8)
    max_iterations = target_tokens * 4 + 64  # generous safety bound
    iterations = 0
    while n_tokens < target_tokens and iterations < max_iterations:
        words.extend(rng.choice(_FILLER_VOCAB) for _ in range(coarse_batch))
        text = " ".join(words)
        n_tokens = len(encoder.tokenizer(text, truncation=False)["input_ids"])
        iterations += 1

    # Fine phase: add one word at a time until we cross the target exactly.
    while n_tokens < target_tokens and iterations < max_iterations:
        words.append(rng.choice(_FILLER_VOCAB))
        text = " ".join(words)
        n_tokens = len(encoder.tokenizer(text, truncation=False)["input_ids"])
        iterations += 1

    if n_tokens < target_tokens:
        raise RuntimeError(
            f"Could not build filler text reaching {target_tokens} tokens "
            f"after {iterations} iterations (reached {n_tokens})."
        )
    return " ".join(words)


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
    text = build_filler_text(encoder, max_len)
    texts = [text] * args.batch_size
    actual_lengths = encoder.token_lengths(texts)
    real_tokens = sum(actual_lengths)
    # Identical strings -> identical truncated lengths; take the first as the
    # representative real (non-pad) sequence length.
    seq_len_actual = actual_lengths[0] if actual_lengths else 0

    warning: str | None = None
    if max_len > 0 and seq_len_actual < SEQ_LEN_WARN_THRESHOLD * max_len:
        warning = (
            f"seq_len_actual={seq_len_actual} is less than "
            f"{SEQ_LEN_WARN_THRESHOLD:.0%} of requested seq_len={max_len}; "
            "this benchmark run is NOT measuring the requested sequence length."
        )
        logger.warning(warning)

    # Measure the real padded tensor width by tokenizing the batch exactly as
    # encode_batch will (same padding mode, same truncation/max_length).
    pad_mode = args.padding
    probe = encoder.tokenizer(
        texts,
        padding=pad_mode,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    padded_width = int(probe["input_ids"].shape[1])

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

    padded_tokens = args.batch_size * padded_width * args.steps
    real_total_tokens = real_tokens * args.steps
    elapsed_s = sum(step_ms) / 1000.0
    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "batch_size": args.batch_size,
        "seq_len": max_len,
        "seq_len_actual": seq_len_actual,
        "padded_width": padded_width,
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
        "warning": warning,
    }
    return result


def _run_sweep_point(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    """Run one sweep point, converting an OOM into a recorded result.

    Previously an OOM at any sweep batch size aborted the entire sweep and lost
    every earlier (successful) point. It also relied on a single
    empty_cache()/gc.collect() at the very end of main() to reclaim memory
    between points, which does nothing while the sweep loop itself is running.
    """
    point_args = argparse.Namespace(**{**vars(args), "batch_size": batch_size})
    try:
        return run_benchmark(point_args)
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("OOM at batch_size=%d: %s", batch_size, exc)
        return {"batch_size": batch_size, "oom": True, "error": str(exc)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


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
        results = [_run_sweep_point(args, bs) for bs in batch_sizes]
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
