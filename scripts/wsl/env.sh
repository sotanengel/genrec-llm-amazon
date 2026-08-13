#!/usr/bin/env bash
set -euo pipefail

# scripts/wsl/env.sh
#
# Sourced by every WSL wrapper script (bootstrap.sh, doctor.sh, fetch_models.sh,
# run.sh). Centralizes the environment policy learned from prior WSL2 + RTX 3060 Ti
# (driver 610.47) incidents on this machine. See README.md's WSL section for the
# full incident writeups.
#
# Usage: source this file, do not execute it directly.
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- Guard against expandable_segments + VRAM oversubscription (WSL2 dxgkrnl bug) ---
# On this machine (WSL2, NVIDIA driver 610.47), PYTORCH_CUDA_ALLOC_CONF=
# expandable_segments:True combined with VRAM oversubscription causes a delayed,
# misleading "RuntimeError: CUDA driver error: device not ready" at a random op
# (confirmed via `dmesg | grep make_resident` showing
# `dxgkio_make_resident: Ioctl failed: -12`, i.e. ENOMEM). Root cause:
# expandable_segments uses the CUDA VMM API (cuMemCreate/cuMemMap), which takes a
# different residency path through the dxgkrnl paravirtualization layer than
# ordinary cudaMalloc; that path is broken under WSL2 on this driver. NEVER set
# this variable on this machine -- unset it and fail loudly if it survives.
unset PYTORCH_CUDA_ALLOC_CONF
if [ -n "${PYTORCH_CUDA_ALLOC_CONF+x}" ]; then
  echo "FATAL: PYTORCH_CUDA_ALLOC_CONF is still set to '${PYTORCH_CUDA_ALLOC_CONF}' after unset." >&2
  echo "       This variable is known to crash CUDA on WSL2 + driver 610.47 when VRAM is" >&2
  echo "       oversubscribed (see scripts/wsl/env.sh for the root cause). Something in your" >&2
  echo "       shell startup (.bashrc/.profile/a wrapper script) is re-exporting it -- remove it." >&2
  return 1 2>/dev/null || exit 1
fi

# --- WSL CUDA stub library path ---
# WSL's /usr/lib/wsl/lib provides libcuda.so.1 (the paravirtualized stub backed by
# the Windows host driver) and the WSL-aware nvidia-smi. NEVER install a Linux
# NVIDIA driver package (nvidia-driver-*, cuda-drivers) inside WSL -- doing so
# overwrites this stub and is the single most common cause of "CUDA not available"
# on WSL2.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# /etc/wsl.conf sets [interop] appendWindowsPath=false for exec-time speed (see
# install_distro.ps1); make sure /usr/lib/wsl/lib (where nvidia-smi lives) stays
# reachable regardless.
export PATH="/usr/lib/wsl/lib:${HOME}/.local/bin:${PATH:-}"

# --- Hugging Face cache: must live on ext4, never /mnt/c. ---
# /mnt/c is a 9p mount and is 10-50x slower for many-small-file / mmap-backed
# workloads (safetensors, tokenizer.json, dataset shards). doctor.sh asserts this.
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

# HF_TOKEN: read from a local, chmod-600, never-committed file rather than baking
# it into any script or the repo. Unauthenticated Hub access is rate-limited and
# flaky, so set this before large downloads (see fetch_models.sh).
HF_TOKEN_FILE="${HOME}/.config/genrec/hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
  export HF_TOKEN
fi

# HF_HUB_OFFLINE is intentionally NOT exported here. Once models have been fetched
# once (scripts/wsl/fetch_models.sh), run.sh --offline turns this on for
# reproducible, network-independent runs.

# Tokenizer parallelism: encode/verbalize run single-process (no fork), so leave
# this "true" for full tokenizer throughput. NOTE: the pytest path wants
# TOKENIZERS_PARALLELISM=false instead, because pytest-xdist forks worker
# processes and tokenizers' parallelism + fork combination deadlocks/warns; that
# override belongs in the test invocation (e.g. Makefile's test-fast target or
# CI env), not here.
export TOKENIZERS_PARALLELISM=true

# Thread counts match .wslconfig's [wsl2] processors=8.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# torch.use_deterministic_algorithms(True) requires CUBLAS_WORKSPACE_CONFIG to be
# set for some cuBLAS ops. It has a (usually small) perf/memory cost, so only set
# it when a caller explicitly opts into determinism.
if [ "${GENREC_DETERMINISTIC:-0}" = "1" ]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
