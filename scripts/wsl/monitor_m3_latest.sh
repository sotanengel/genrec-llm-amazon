#!/usr/bin/env bash
# Print the latest one-line monitor summary (for agent/user polling).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUMMARY="${ROOT}/logs/monitor_summary.log"
bash "${ROOT}/scripts/wsl/monitor_m3_production.sh" --once >/dev/null
if [[ ! -f "${SUMMARY}" ]]; then
  echo "no monitor_summary.log yet; run: bash scripts/wsl/monitor_m3_production.sh"
  exit 1
fi
tail -1 "${SUMMARY}"
