#!/usr/bin/env bash
# Monitor M3 production pipeline. Writes one-line summaries to logs/monitor_summary.log
# Schedule: 1 min x 10, then every 30 min until pipeline finishes.
# Usage: bash scripts/wsl/monitor_m3_production.sh [--once]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

SUMMARY="${ROOT}/logs/monitor_summary.log"
SNAP="${ROOT}/logs/monitor_snapshot.log"
LOCK="${ROOT}/logs/monitor_m3.lock"
INTERVAL_SLOW="${MONITOR_INTERVAL_SLOW:-1800}"
COMPLETE_MARKER="${ROOT}/logs/state/m3_resume.complete"
FAILED_MARKER="${ROOT}/logs/state/m3_resume.failed"
SERVICE_NAME="genrec-m3-resume.service"

service_active() {
  systemctl --user is-active --quiet "${SERVICE_NAME}" >/dev/null 2>&1
}

service_failed() {
  systemctl --user is-failed --quiet "${SERVICE_NAME}" >/dev/null 2>&1
}

pipeline_alive() {
  service_active \
    || pgrep -f 'run_m3_production.sh|scripts/run_m3.sh|resume_m3_train_head.sh|genrec_lite train head|genrec_lite report build' >/dev/null 2>&1
}

get_phase() {
  if [[ -f "${COMPLETE_MARKER}" ]]; then echo "completed"
  elif pgrep -f 'data prepare' >/dev/null 2>&1; then echo "data-prepare"
  elif pgrep -f 'encode run.*scope eval' >/dev/null 2>&1; then echo "encode-eval"
  elif pgrep -f 'encode run.*scope train' >/dev/null 2>&1; then echo "encode-train"
  elif pgrep -f 'train head' >/dev/null 2>&1; then echo "train-head"
  elif pgrep -f 'report build' >/dev/null 2>&1; then echo "report"
  elif pgrep -f 'resume_m3_train_head.sh' >/dev/null 2>&1; then echo "resume-shell"
  elif service_active; then echo "systemd-resume"
  elif service_failed || [[ -f "${FAILED_MARKER}" ]]; then echo "failed"
  elif pgrep -f 'run_m3_production.sh|scripts/run_m3.sh' >/dev/null 2>&1; then echo "pipeline-shell"
  else echo "stopped"
  fi
}

get_log_tail() {
  local tee_pid latest
  tee_pid=$(pgrep -f 'tee -a.*run_m3_' | head -1 || true)
  if [[ -n "${tee_pid}" ]] && [[ -r "/proc/${tee_pid}/fd/3" ]]; then
    tail -3 "/proc/${tee_pid}/fd/3" 2>/dev/null | tr '\n' ' '
  else
    latest=$(ls -t logs/resume_train_head_*.log logs/run_m3_*.log 2>/dev/null | head -1 || true)
    if [[ -n "${latest}" ]]; then tail -3 "${latest}" | tr '\n' ' '; else echo "no-log"; fi
  fi
}

is_complete() {
  [[ -f "${COMPLETE_MARKER}" ]] \
    || {
      [[ -f "${ROOT}/reports/results.md" ]] \
        && grep -q '==> done' logs/run_m3_*.log 2>/dev/null
    }
}

check_once() {
  local phase gpu proc_count memmap_bytes line
  phase=$(get_phase)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
  proc_count=$(pgrep -cf 'run_m3|resume_m3|genrec_lite' 2>/dev/null || echo 0)
  memmap_bytes=$(stat -c%s cache/hidden_states/*.train.f16.memmap 2>/dev/null || echo 0)
  line="$(date -Iseconds) | phase=${phase} | procs=${proc_count} | gpu=${gpu} | train_memmap=${memmap_bytes}B | $(get_log_tail)"
  echo "${line}" | tee -a "${SUMMARY}"
  {
    echo "========== $(date -Iseconds) =========="
    ps aux | grep -E 'run_m3|resume_m3|genrec_lite' | grep -v grep || echo "[monitor] no pipeline processes"
    echo "--- summary ---"
    echo "${line}"
    echo ""
  } >> "${SNAP}"
}

run_loop() {
  for i in $(seq 1 10); do
    check_once
    is_complete && return 0
    [[ $i -lt 10 ]] && sleep 60
  done
  while pipeline_alive; do
    sleep "${INTERVAL_SLOW}"
    check_once
    is_complete && return 0
  done
  check_once
  if is_complete; then
    echo "$(date -Iseconds) | phase=completed | durable completion marker present" | tee -a "${SUMMARY}"
  elif service_failed || [[ -f "${FAILED_MARKER}" ]]; then
    echo "$(date -Iseconds) | phase=failed | inspect systemd journal and failure marker" | tee -a "${SUMMARY}"
    return 1
  else
    echo "$(date -Iseconds) | phase=stopped | pipeline processes gone" | tee -a "${SUMMARY}"
  fi
}

if [[ "${1:-}" == "--once" ]]; then
  check_once
  exit 0
fi

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "monitor already running (lock: ${LOCK})" >&2
  exit 0
fi

echo "$$" >"${LOCK}.pid"
trap 'rm -f "${LOCK}.pid"' EXIT

run_loop
