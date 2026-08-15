#!/usr/bin/env bash
# Production M3 pipeline (Qwen3-8B-Base nf4, amazon_video_games).
# Usage: bash scripts/wsl/run_m3_production.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:/usr/bin:/bin:${PATH:-}"
export GENREC_DATASET="${GENREC_DATASET:-amazon_video_games}"
export GENREC_MODEL="${GENREC_MODEL:-qwen3-8b-base}"
export GENREC_VERBALIZER="${GENREC_VERBALIZER:-v1_full}"
export GENREC_EXP="${GENREC_EXP:-m3_frozen}"

LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_m3_$(date +%Y%m%d_%H%M%S).log"

echo "==> M3 production run" | tee -a "${LOG_FILE}"
echo "    dataset=${GENREC_DATASET} model=${GENREC_MODEL} verbalizer=${GENREC_VERBALIZER} exp=${GENREC_EXP}" | tee -a "${LOG_FILE}"

if ! bash scripts/wsl/run.sh bash scripts/run_m3.sh 2>&1 | tee -a "${LOG_FILE}"; then
  echo "[ERROR] run_m3.sh failed. See ${LOG_FILE}" >&2
  exit 1
fi

echo "==> finished OK. log: ${LOG_FILE}"
