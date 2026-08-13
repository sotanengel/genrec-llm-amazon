#!/usr/bin/env bash
set -euo pipefail

# scripts/wsl/doctor.sh
#
# The definition of "WSL2 GPU bring-up complete" for this repo. Runs every check
# independently (never chains fallible commands with ';', which would only
# propagate the last exit code), prints a PASS/FAIL table, and exits non-zero if
# any check failed.
#
# Usage:
#   bash scripts/wsl/doctor.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_DIR}"

declare -a RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0

record() {
  local name="$1" status="$2" detail="${3:-}"
  RESULTS+=("${status}|${name}|${detail}")
  if [ "${status}" = "PASS" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

# --- 1. WSL version / kernel -------------------------------------------------
KREL="$(uname -r)"
if printf '%s' "${KREL}" | grep -qi "microsoft"; then
  if printf '%s' "${KREL}" | grep -qi "WSL2"; then
    record "WSL version/kernel" "PASS" "kernel ${KREL}"
  else
    record "WSL version/kernel" "FAIL" "kernel ${KREL} does not look like WSL2 (missing 'WSL2' tag; WSL1?)"
  fi
else
  record "WSL version/kernel" "FAIL" "not running under WSL (uname -r: ${KREL})"
fi

# --- 2. libcuda.so.1 stub -----------------------------------------------------
if [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
  record "libcuda.so.1 stub" "PASS" "/usr/lib/wsl/lib/libcuda.so.1"
else
  record "libcuda.so.1 stub" "FAIL" "missing -- is the Windows NVIDIA driver installed with WSL GPU support enabled?"
fi

# --- 3. nvidia-smi -------------------------------------------------------------
NVIDIA_SMI_OUT="$(mktemp)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >"${NVIDIA_SMI_OUT}" 2>&1; then
  record "nvidia-smi" "PASS" "$(head -n1 "${NVIDIA_SMI_OUT}")"
else
  record "nvidia-smi" "FAIL" "nvidia-smi not found or failed -- check PATH/LD_LIBRARY_PATH include /usr/lib/wsl/lib"
fi
rm -f "${NVIDIA_SMI_OUT}"

# --- 4-7. torch/CUDA facts (single python probe; only reachable if a venv with
# torch exists, i.e. bootstrap.sh has been run) --------------------------------
if command -v uv >/dev/null 2>&1; then
  PYRUN=(uv run --project "${REPO_DIR}" --frozen python)
else
  PYRUN=(python3)
fi

TORCH_PROBE_OUT="$(mktemp)"
TORCH_PROBE_ERR="$(mktemp)"
set +e
"${PYRUN[@]}" - <<'PYEOF' >"${TORCH_PROBE_OUT}" 2>"${TORCH_PROBE_ERR}"
try:
    import torch
except Exception as e:  # noqa: BLE001 -- diagnostic probe, report any import failure
    print(f"RESULT|torch import|FAIL|{e}")
    print("RESULT|torch.cuda.is_available()|FAIL|torch not importable")
    print("RESULT|torch.cuda.get_device_capability()==(8,6)|FAIL|torch not importable")
    print("RESULT|torch.cuda.is_bf16_supported()|FAIL|torch not importable")
    print("RESULT|free VRAM|FAIL|torch not importable")
    raise SystemExit(0)

print(f"RESULT|torch import|PASS|{torch.__version__}")

avail = torch.cuda.is_available()
print(f"RESULT|torch.cuda.is_available()|{'PASS' if avail else 'FAIL'}|{avail}")

if avail:
    cap = tuple(torch.cuda.get_device_capability())
    ok = cap == (8, 6)
    print(f"RESULT|torch.cuda.get_device_capability()==(8,6)|{'PASS' if ok else 'FAIL'}|{cap}")

    bf16 = torch.cuda.is_bf16_supported()
    print(f"RESULT|torch.cuda.is_bf16_supported()|{'PASS' if bf16 else 'FAIL'}|{bf16}")

    free, total = torch.cuda.mem_get_info()
    free_mib = free / (1024 * 1024)
    total_mib = total / (1024 * 1024)
    print(f"RESULT|free VRAM|PASS|{free_mib:.0f} MiB free / {total_mib:.0f} MiB total")
else:
    print("RESULT|torch.cuda.get_device_capability()==(8,6)|FAIL|CUDA not available")
    print("RESULT|torch.cuda.is_bf16_supported()|FAIL|CUDA not available")
    print("RESULT|free VRAM|FAIL|CUDA not available")
PYEOF
PROBE_STATUS=$?
set -e

if [ -s "${TORCH_PROBE_OUT}" ]; then
  while IFS='|' read -r tag name status detail; do
    [ "${tag}" = "RESULT" ] || continue
    record "${name}" "${status}" "${detail}"
  done <"${TORCH_PROBE_OUT}"
else
  ERR_HEAD="$(head -n1 "${TORCH_PROBE_ERR}" 2>/dev/null || true)"
  record "torch import" "FAIL" "probe produced no output (exit ${PROBE_STATUS}): ${ERR_HEAD}. Run scripts/wsl/bootstrap.sh first."
  record "torch.cuda.is_available()" "FAIL" "probe failed"
  record "torch.cuda.get_device_capability()==(8,6)" "FAIL" "probe failed"
  record "torch.cuda.is_bf16_supported()" "FAIL" "probe failed"
  record "free VRAM" "FAIL" "probe failed"
fi
rm -f "${TORCH_PROBE_OUT}" "${TORCH_PROBE_ERR}"

# --- 8. HF_HOME is on ext4, never /mnt/c (9p) ----------------------------------
if [ -n "${HF_HOME:-}" ]; then
  mkdir -p "${HF_HOME}" 2>/dev/null || true
  FS_TYPE="$(stat -f -c %T "${HF_HOME}" 2>/dev/null || echo "unknown")"
  case "${FS_TYPE}" in
    v9fs | 9p)
      record "HF_HOME on ext4" "FAIL" "HF_HOME=${HF_HOME} is on ${FS_TYPE} (i.e. /mnt/c) -- move it to an ext4 path, e.g. \$HOME/.cache/huggingface"
      ;;
    unknown)
      record "HF_HOME on ext4" "FAIL" "could not determine filesystem type for HF_HOME=${HF_HOME} (stat -f unsupported here?)"
      ;;
    *)
      record "HF_HOME on ext4" "PASS" "HF_HOME=${HF_HOME} on ${FS_TYPE}"
      ;;
  esac
else
  record "HF_HOME on ext4" "FAIL" "HF_HOME is not set (expected env.sh to set it)"
fi

# --- 9. PYTORCH_CUDA_ALLOC_CONF unset -------------------------------------------
if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  record "PYTORCH_CUDA_ALLOC_CONF unset" "PASS" ""
else
  record "PYTORCH_CUDA_ALLOC_CONF unset" "FAIL" "set to '${PYTORCH_CUDA_ALLOC_CONF}' -- known to crash CUDA on this machine, see env.sh"
fi

# --- 10. MemTotal ~= 24 GB and <=75% of Windows host physical RAM --------------
MEM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
MEM_TOTAL_GB="$(awk -v kb="${MEM_TOTAL_KB}" 'BEGIN { printf "%.1f", kb / 1024 / 1024 }')"

if awk -v g="${MEM_TOTAL_GB}" 'BEGIN { exit !(g >= 20 && g <= 26) }'; then
  record "WSL MemTotal ~24GB" "PASS" "${MEM_TOTAL_GB} GiB"
else
  record "WSL MemTotal ~24GB" "FAIL" "${MEM_TOTAL_GB} GiB (expected ~24 GiB per %USERPROFILE%\\.wslconfig [wsl2] memory=24GB)"
fi

# Host physical RAM: try Windows interop first, fall back to the measured value
# documented in README.md (override with GENREC_HOST_RAM_GB if hardware changed).
HOST_RAM_GB="${GENREC_HOST_RAM_GB:-}"
HOST_RAM_SOURCE="GENREC_HOST_RAM_GB override"
if [ -z "${HOST_RAM_GB}" ]; then
  PWSH="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
  if [ -x "${PWSH}" ]; then
    HOST_RAM_GB="$("${PWSH}" -NoProfile -Command \
      '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)' \
      2>/dev/null | tr -d '\r' || true)"
    HOST_RAM_SOURCE="Windows interop (Get-CimInstance Win32_ComputerSystem)"
  fi
fi
if [ -z "${HOST_RAM_GB}" ]; then
  HOST_RAM_GB="31.8"
  HOST_RAM_SOURCE="documented fallback (measured 2026-07; interop unavailable)"
fi

if awk -v m="${MEM_TOTAL_GB}" -v h="${HOST_RAM_GB}" 'BEGIN { exit !(m <= h * 0.75 + 0.1) }'; then
  RATIO="$(awk -v m="${MEM_TOTAL_GB}" -v h="${HOST_RAM_GB}" 'BEGIN { printf "%.1f%%", m / h * 100 }')"
  record ".wslconfig memory <=75% of host RAM" "PASS" "${MEM_TOTAL_GB}/${HOST_RAM_GB} GiB = ${RATIO} (host RAM via ${HOST_RAM_SOURCE})"
else
  record ".wslconfig memory <=75% of host RAM" "FAIL" "${MEM_TOTAL_GB} GiB exceeds 75% of ${HOST_RAM_GB} GiB host RAM (via ${HOST_RAM_SOURCE}) -- risk of Wsl/Service/E_UNEXPECTED crashes, do not raise [wsl2] memory in .wslconfig"
fi

# --- print PASS/FAIL table ------------------------------------------------------
printf '\n%-42s %-6s %s\n' "CHECK" "STATUS" "DETAIL"
printf '%s\n' "--------------------------------------------------------------------------------"
for row in "${RESULTS[@]}"; do
  IFS='|' read -r status name detail <<<"${row}"
  printf '%-42s %-6s %s\n' "${name}" "${status}" "${detail}"
done
printf '\n%d passed, %d failed\n' "${PASS_COUNT}" "${FAIL_COUNT}"

if [ "${FAIL_COUNT}" -gt 0 ]; then
  exit 1
fi
exit 0
