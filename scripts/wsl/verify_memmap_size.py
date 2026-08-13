"""Verify hidden-state memmap byte size matches N * hidden_dim * 2."""

from __future__ import annotations

from genrec_lite.config import (
    find_project_root,
    load_data_config,
    load_llm_config,
    load_verbalizer_config,
)
from genrec_lite.data.schema import read_parquet_bundle
from genrec_lite.encode.cache import CacheKeyConfig, HiddenStateCache, compute_cache_key
from genrec_lite.encode.prefill import ENCODER_VERSION, PrefillEncoder


def main() -> None:
    root = find_project_root()
    data_config = load_data_config("amazon_video_games", config_dir=root / "configs")
    llm_config = load_llm_config("qwen3-1.7b-base", config_dir=root / "configs")
    verb_config = load_verbalizer_config("v1_full", config_dir=root / "configs")
    _, _, _, samples = read_parquet_bundle(root / data_config.output_dir)
    n_samples = samples.height

    encoder = PrefillEncoder.from_config(llm_config)
    hidden_dim = int(encoder.encode_batch(["probe"]).shape[1])
    cache_key = compute_cache_key(
        CacheKeyConfig(
            model_id=llm_config.model_id,
            revision=llm_config.revision,
            verbalizer_name="v1_full",
            verbalizer_config=verb_config.model_dump(),
            max_len=llm_config.max_len,
            dtype=llm_config.dtype,
            quantize=llm_config.quantize,
            pooling=llm_config.pooling,
            attn_implementation=llm_config.attn_implementation,
            deterministic=llm_config.deterministic,
            encoder_version=ENCODER_VERSION,
        )
    )
    cache = HiddenStateCache(root / "cache/hidden_states", cache_key, n_samples, hidden_dim)
    actual = cache.memmap_path.stat().st_size
    expected = cache.expected_bytes
    print(
        f"n={n_samples} hidden_dim={hidden_dim} "
        f"expected_bytes={expected} actual_bytes={actual} ok={actual == expected}"
    )
    if actual != expected:
        raise SystemExit(f"memmap size mismatch: {actual} != {expected}")


if __name__ == "__main__":
    main()
