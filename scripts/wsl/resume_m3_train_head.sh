#!/usr/bin/env bash
# Resume M3 from train head (encode caches already finalized).
# Usage: bash scripts/wsl/resume_m3_train_head.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:/usr/bin:/bin:${PATH:-}"
export GENREC_EXP="${GENREC_EXP:-m3_frozen}"

LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resume_train_head_$(date +%Y%m%d_%H%M%S).log"
STATE_DIR="${LOG_DIR}/state"
COMPLETE_MARKER="${STATE_DIR}/m3_resume.complete"
FAILED_MARKER="${STATE_DIR}/m3_resume.failed"
LOCK_FILE="${STATE_DIR}/m3_resume.lock"
mkdir -p "${STATE_DIR}"

if [[ -f "${COMPLETE_MARKER}" ]] && [[ "${GENREC_FORCE_RESUME:-0}" != "1" ]]; then
  echo "M3 train-head/report already completed: ${COMPLETE_MARKER}"
  exit 0
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "M3 train-head/report resume is already running (lock: ${LOCK_FILE})" >&2
  exit 0
fi

rm -f "${FAILED_MARKER}"
exec > >(tee -a "${LOG_FILE}") 2>&1

record_failure() {
  local status=$?
  printf 'failed_at=%s\nexit_status=%s\nlog=%s\n' \
    "$(date -Iseconds)" "${status}" "${LOG_FILE}" >"${FAILED_MARKER}.tmp"
  mv "${FAILED_MARKER}.tmp" "${FAILED_MARKER}"
  exit "${status}"
}
trap record_failure ERR

if [[ "${GENREC_SKIP_MONITOR:-}" != "1" ]]; then
  nohup bash scripts/wsl/monitor_m3_production.sh >> "${LOG_DIR}/monitor_nohup.out" 2>&1 &
fi

echo "==> resume train head + report"
echo "    exp=${GENREC_EXP} started=$(date -Iseconds)"
bash scripts/wsl/run.sh python -m genrec_lite train head --exp "${GENREC_EXP}"
echo "==> report build"
bash scripts/wsl/run.sh python -m genrec_lite report build --exp "${GENREC_EXP}"
echo "==> done $(date -Iseconds)"

printf 'completed_at=%s\nexp=%s\nlog=%s\n' \
  "$(date -Iseconds)" "${GENREC_EXP}" "${LOG_FILE}" >"${COMPLETE_MARKER}.tmp"
mv "${COMPLETE_MARKER}.tmp" "${COMPLETE_MARKER}"
rm -f "${FAILED_MARKER}"
trap - ERR

echo "==> finished OK. log: ${LOG_FILE}"
