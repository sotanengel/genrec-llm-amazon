#!/usr/bin/env python3
"""Check GPU VRAM usage for a prefill encoder (issue #11).

Correctness note (measurement bug fix, 2026-08-14): this script had two
defects that made its measured VRAM numbers unreliable for a requested
``--seq-len``:

1. Filler text was built from a fixed "16 characters per token" guess
   (``"vram check prompt " * (seq_len // 16)``), which for real tokenizers
   produced sequences far shorter than the requested length -- the same class
   of bug found in ``bench_prefill.py``. Since peak VRAM scales with the real
   sequence length, this understated the true cost of the seq-len it claimed
   to measure. Fixed below with a filler builder (mirroring
   ``bench_prefill.build_filler_text``) that grows deterministic, varied
   filler text and re-tokenizes it until it verifiably reaches the target
   token count -- no fixed ratio assumed. (Duplicated rather than imported:
   scripts/ is not a package and is not on mypy's module search path, so a
   cross-script import isn't resolvable; the function is small enough that
   duplication is simpler than restructuring the build.)
2. ``--seq-len`` was accepted on the CLI but never actually applied: the
   encoder was built via ``PrefillEncoder.from_config(llm_cfg, ...)``, which
   always used the *config file's* ``max_len`` for truncation, so changing
   ``--seq-len`` had **no effect** on what was measured -- every run silently
   measured at the config's default max_len regardless of the flag. Likewise
   ``--dtype``/``--quantize`` were accepted but never forwarded to the
   encoder. Fixed by constructing ``PrefillEncoder`` directly with the
   CLI-provided ``seq_len``/``dtype``/``quantize`` (falling back to the
   config's values when not given), matching ``bench_prefill.py``'s pattern.
"""

from __future__ import annotations

import argparse
import gc
import logging
import random

import torch

from genrec_lite.config import find_project_root, load_llm_config
from genrec_lite.encode.prefill import PrefillEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Minimum fraction of the requested --seq-len that the achieved (real,
# truncated) token count must reach before this is considered to actually be
# measuring what it claims. Kept in sync with bench_prefill.py's threshold.
SEQ_LEN_WARN_THRESHOLD = 0.95

# Varied vocabulary for filler text, matching bench_prefill.py: a single
# repeated token is an unrepresentative attention workload.
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


def build_filler_text(encoder: PrefillEncoder, target_tokens: int, seed: int = 0) -> str:
    """Build varied, deterministic filler text that tokenizes to >= target_tokens.

    See bench_prefill.build_filler_text for the full rationale; kept in sync
    with that implementation.
    """
    if target_tokens <= 0:
        return "vram check prompt"

    rng = random.Random(seed)
    words: list[str] = []
    n_tokens = 0

    coarse_batch = max(1, target_tokens // 8)
    max_iterations = target_tokens * 4 + 64
    iterations = 0
    while n_tokens < target_tokens and iterations < max_iterations:
        words.extend(rng.choice(_FILLER_VOCAB) for _ in range(coarse_batch))
        text = " ".join(words)
        n_tokens = len(encoder.tokenizer(text, truncation=False)["input_ids"])
        iterations += 1

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
    encoder = PrefillEncoder(
        model_id=llm_cfg.model_id,
        dtype=dtype or llm_cfg.dtype,
        pooling=llm_cfg.pooling,
        max_len=seq_len,
        quantize=quantize or llm_cfg.quantize,
        device="cuda",
        revision=llm_cfg.revision,
        attn_implementation=llm_cfg.attn_implementation,
        low_cpu_mem_usage=llm_cfg.low_cpu_mem_usage,
        bnb_compute_dtype=llm_cfg.bnb_compute_dtype,
        trust_remote_code=llm_cfg.trust_remote_code,
        deterministic=llm_cfg.deterministic,
    )
    text = build_filler_text(encoder, seq_len)
    texts = [text] * batch_size
    actual_lengths = encoder.token_lengths(texts)
    seq_len_actual = actual_lengths[0] if actual_lengths else 0
    if seq_len > 0 and seq_len_actual < SEQ_LEN_WARN_THRESHOLD * seq_len:
        logger.warning(
            "seq_len_actual=%d is below %.0f%% of requested seq_len=%d; "
            "this run is NOT measuring the requested sequence length.",
            seq_len_actual,
            SEQ_LEN_WARN_THRESHOLD * 100,
            seq_len,
        )

    encoder.encode_batch(texts)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    logger.info(
        "batch=%d seq_requested=%d seq_actual=%d peak=%.2f GB reserved=%.2f GB",
        batch_size,
        seq_len,
        seq_len_actual,
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
