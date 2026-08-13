#!/usr/bin/env bash
set -euo pipefail

# scripts/wsl/run.sh
#
# Universal entrypoint for running anything in this repo's uv-managed
# environment with the WSL env policy from env.sh applied.
#
# Usage:
#   scripts/wsl/run.sh [--offline] <command> [args...]
#
# Examples:
#   scripts/wsl/run.sh python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base
#   scripts/wsl/run.sh pytest -x -n auto tests/
#   scripts/wsl/run.sh --offline python -m genrec_lite eval run --exp m1_baselines

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

OFFLINE=0
if [ "${1:-}" = "--offline" ]; then
  OFFLINE=1
  shift
fi

if [ "${OFFLINE}" -eq 1 ]; then
  export HF_HUB_OFFLINE=1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $(basename "${BASH_SOURCE[0]}") [--offline] <command> [args...]" >&2
  exit 2
fi

cd "${REPO_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv not found on PATH. Run scripts/wsl/bootstrap.sh first." >&2
  exit 1
fi

exec uv run --frozen "$@"
