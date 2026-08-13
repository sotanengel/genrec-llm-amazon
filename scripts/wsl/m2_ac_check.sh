#!/usr/bin/env bash
# M2 acceptance checks for 1.7B prefill encode (DESIGN.md §2.3.3 / issue #5).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
# shellcheck source=scripts/wsl/env.sh
source scripts/wsl/env.sh

log() { printf '[m2-ac] %s\n' "$*" >&2; }

log "Step 1/5: prepare synthetic mini dataset"
uv run --frozen python scripts/wsl/prepare_m2_mini.py

log "Step 2/5: encode run (1.7B bf16, non-deterministic)"
rm -rf cache/hidden_states
uv run --frozen python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base

log "Step 3/5: memmap size == N * d * 2"
uv run --frozen python scripts/wsl/verify_memmap_size.py

log "Step 4/5: deterministic encode x2 -> sha256 match"
export GENREC_DETERMINISTIC=1
rm -rf cache/hidden_states
uv run --frozen python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base
HASH1="$(sha256sum cache/hidden_states/*.f16.memmap | awk '{print $1}')"
rm -rf cache/hidden_states
uv run --frozen python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base
HASH2="$(sha256sum cache/hidden_states/*.f16.memmap | awk '{print $1}')"
if [ "${HASH1}" != "${HASH2}" ]; then
  log "FATAL: deterministic hashes differ: ${HASH1} vs ${HASH2}"
  exit 1
fi
log "deterministic sha256 OK: ${HASH1}"

log "Step 5/5: test-gpu"
uv run --frozen pytest tests/test_gpu_prefill.py -m gpu -q
log "M2 AC checks PASSED"
