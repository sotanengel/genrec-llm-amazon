"""Verify hidden-state memmap byte size matches N * hidden_dim * 2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from genrec_lite.config import (
    find_project_root,
    load_data_config,
    load_llm_config,
    load_verbalizer_config,
)
from genrec_lite.data.schema import read_parquet_bundle, read_train_samples
from genrec_lite.encode.cache import CacheKeyConfig, CacheScope, HiddenStateCache, compute_cache_key
from genrec_lite.encode.prefill import ENCODER_VERSION, PrefillEncoder


def _encode_deterministic(llm_config: object) -> bool:
    deterministic = getattr(llm_config, "deterministic", False)
    return bool(deterministic) or os.environ.get("GENREC_DETERMINISTIC") == "1"


def resolve_cache_key(
    *,
    model: str,
    verbalizer: str,
    root: Path,
) -> str:
    llm_config = load_llm_config(model, config_dir=root / "configs")
    verb_config = load_verbalizer_config(verbalizer, config_dir=root / "configs")
    deterministic = _encode_deterministic(llm_config)
    return compute_cache_key(
        CacheKeyConfig(
            model_id=llm_config.model_id,
            revision=llm_config.revision,
            verbalizer_name=verbalizer,
            verbalizer_config=verb_config.model_dump(),
            max_len=llm_config.max_len,
            dtype=llm_config.dtype,
            quantize=llm_config.quantize,
            pooling=llm_config.pooling,
            attn_implementation=llm_config.attn_implementation,
            deterministic=deterministic,
            encoder_version=ENCODER_VERSION,
        )
    )


def verify_memmap_size(
    *,
    dataset: str,
    model: str,
    verbalizer: str,
    cache_dir: str = "cache/hidden_states",
    hidden_dim: int | None = None,
    scope: CacheScope = "eval",
    root: Path | None = None,
) -> None:
    project_root = root or find_project_root()
    data_config = load_data_config(dataset, config_dir=project_root / "configs")
    llm_config = load_llm_config(model, config_dir=project_root / "configs")
    data_path = project_root / data_config.output_dir
    if scope == "train":
        n_samples = read_train_samples(data_path).height
    else:
        _, _, _, samples = read_parquet_bundle(data_path)
        n_samples = samples.height

    if hidden_dim is None:
        encoder = PrefillEncoder.from_config(llm_config)
        hidden_dim = int(encoder.encode_batch(["probe"]).shape[1])

    cache_key = resolve_cache_key(
        model=model,
        verbalizer=verbalizer,
        root=project_root,
    )
    cache = HiddenStateCache(
        project_root / cache_dir,
        cache_key,
        n_samples,
        hidden_dim,
        scope=scope,
    )
    actual = cache.memmap_path.stat().st_size
    expected = cache.expected_bytes
    print(
        f"dataset={dataset} model={model} verbalizer={verbalizer} scope={scope} "
        f"n={n_samples} hidden_dim={hidden_dim} "
        f"expected_bytes={expected} actual_bytes={actual} ok={actual == expected}"
    )
    if actual != expected:
        raise SystemExit(f"memmap size mismatch: {actual} != {expected}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="amazon_video_games",
        help="Dataset config name (default: amazon_video_games)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-1.7b-base",
        help="LLM config name (default: qwen3-1.7b-base)",
    )
    parser.add_argument(
        "--verbalizer",
        default="v1_full",
        help="Verbalizer config name (default: v1_full)",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/hidden_states",
        help="Hidden-state cache directory (default: cache/hidden_states)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Skip model load by supplying hidden dimension explicitly",
    )
    parser.add_argument(
        "--scope",
        choices=("eval", "train"),
        default="eval",
        help="eval verifies samples.parquet memmap; train verifies train_samples.parquet",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verify_memmap_size(
        dataset=args.dataset,
        model=args.model,
        verbalizer=args.verbalizer,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
