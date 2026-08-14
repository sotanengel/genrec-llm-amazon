#!/usr/bin/env bash
# Full Amazon Video Games (leave_one_out) encode with Qwen3-8B nf4.
#
# Usage (from repo root, inside WSL ext4 clone):
#   scripts/wsl/run_full_encode_literature.sh
#   scripts/wsl/run_full_encode_literature.sh --prepare-only
#   scripts/wsl/run_full_encode_literature.sh --encode-only
#
# Long runs: wrap with nohup/tmux; encode resumes via cache progress logs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
# shellcheck source=scripts/wsl/env.sh
source scripts/wsl/env.sh

DATASET="amazon_video_games_literature"
MODEL="qwen3-8b-base"
VERBALIZER="v1_full"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/encode_literature_8b.log"

PREPARE_ONLY=0
ENCODE_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --prepare-only) PREPARE_ONLY=1 ;;
    --encode-only) ENCODE_ONLY=1 ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [--prepare-only|--encode-only]" >&2
      exit 2
      ;;
  esac
done

log() { printf '[full-encode] %s\n' "$*" >&2; }

if [ "${ENCODE_ONLY}" -eq 0 ]; then
  log "Step 1/4: data prepare (${DATASET})"
  uv run --frozen python -m genrec_lite data prepare --dataset "${DATASET}"

  log "Step 2/4: data stats (expect ~94762 users / ~25612 items / ~814586 interactions)"
  uv run --frozen python -m genrec_lite data stats --dataset "${DATASET}"
fi

if [ "${PREPARE_ONLY}" -eq 1 ]; then
  log "prepare-only complete"
  exit 0
fi

if [ "${ENCODE_ONLY}" -eq 0 ]; then
  log "Step 3/4: fetch model (${MODEL})"
  bash scripts/wsl/fetch_models.sh "${MODEL}"

  log "Step 3b: verbalizer sample dump (human review)"
  uv run --frozen python -m genrec_lite verbalize render \
    --dataset "${DATASET}" --verbalizer "${VERBALIZER}" --n 20
fi

mkdir -p "${LOG_DIR}"
log "Step 4/4: encode run (${MODEL}, ${VERBALIZER}) -> ${LOG_FILE}"
uv run --frozen python -m genrec_lite encode run \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --verbalizer "${VERBALIZER}" \
  --verbose 2>&1 | tee -a "${LOG_FILE}"

log "Step 5/5: verify memmap size (Qwen3-8B hidden_dim=4096)"
uv run --frozen python scripts/wsl/verify_memmap_size.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --verbalizer "${VERBALIZER}" \
  --hidden-dim 4096

log "Full literature encode pipeline complete"
