#!/usr/bin/env bash
set -euo pipefail

# scripts/wsl/fetch_models.sh
#
# Downloads the LLM(s) declared in configs/model/llm/*.yaml using `hf download`
# (the `huggingface-cli download` command is deprecated in favour of `hf
# download`). Resumable: re-running skips files already cached under HF_HOME.
# configs/model/llm/*.yaml is the single source of truth for model_id/revision;
# this script only reads it, it never writes to it.
#
# Usage:
#   bash scripts/wsl/fetch_models.sh                  # fetch every configs/model/llm/*.yaml
#   bash scripts/wsl/fetch_models.sh qwen3-1.7b-base   # fetch one config, by file stem
#
# After fetching, re-run workloads with:
#   scripts/wsl/run.sh --offline python -m genrec_lite ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_DIR}/configs/model/llm"

# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv not found on PATH. Run scripts/wsl/bootstrap.sh first." >&2
  exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
  echo "FATAL: config dir not found: ${CONFIG_DIR}" >&2
  exit 1
fi

declare -a STEMS=()
if [ "$#" -gt 0 ]; then
  STEMS=("$@")
else
  for f in "${CONFIG_DIR}"/*.yaml; do
    [ -e "${f}" ] || continue
    STEMS+=("$(basename "${f}" .yaml)")
  done
fi

if [ "${#STEMS[@]}" -eq 0 ]; then
  echo "FATAL: no *.yaml configs found under ${CONFIG_DIR}." >&2
  exit 1
fi

FAILED=0
for stem in "${STEMS[@]}"; do
  cfg="${CONFIG_DIR}/${stem}.yaml"
  if [ ! -f "${cfg}" ]; then
    echo "FATAL: no such model config: ${cfg}" >&2
    FAILED=1
    continue
  fi

  echo "== ${stem} (${cfg}) =="

  PARSED="$(mktemp)"
  set +e
  uv run --project "${REPO_DIR}" --frozen python - "${cfg}" >"${PARSED}" 2>&1 <<'PYEOF'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

model_id = cfg.get("model_id", "")
revision = cfg.get("revision", "main")
gated = "1" if cfg.get("gated") else "0"
print(f"{model_id} {revision} {gated}")
PYEOF
  PARSE_STATUS=$?
  set -e

  if [ "${PARSE_STATUS}" -ne 0 ]; then
    echo "FATAL: could not parse ${cfg}:" >&2
    cat "${PARSED}" >&2
    rm -f "${PARSED}"
    FAILED=1
    continue
  fi

  read -r MODEL_ID REVISION GATED <"${PARSED}"
  rm -f "${PARSED}"

  if [ -z "${MODEL_ID}" ]; then
    echo "FATAL: ${cfg} has no model_id." >&2
    FAILED=1
    continue
  fi

  echo "model_id=${MODEL_ID} revision=${REVISION} gated=${GATED}"

  set +e
  uv run --project "${REPO_DIR}" --frozen hf download "${MODEL_ID}" --revision "${REVISION}"
  DL_STATUS=$?
  set -e

  if [ "${DL_STATUS}" -ne 0 ]; then
    if [ "${GATED}" = "1" ]; then
      echo "FATAL: download of gated model '${MODEL_ID}' failed (exit ${DL_STATUS})." >&2
      echo "       Accept the license at https://huggingface.co/${MODEL_ID} while logged" >&2
      echo "       in as the account tied to HF_TOKEN (~/.config/genrec/hf_token), then" >&2
      echo "       re-run this script." >&2
    else
      echo "FATAL: download of '${MODEL_ID}' failed (exit ${DL_STATUS})." >&2
      echo "       Check HF_TOKEN / network connectivity and re-run -- downloads resume" >&2
      echo "       from where they left off." >&2
    fi
    FAILED=1
    continue
  fi
done

if [ "${FAILED}" -ne 0 ]; then
  echo "One or more models failed to download. See FATAL lines above." >&2
  exit 1
fi

echo "All models fetched into HF_HOME=${HF_HOME}."
echo "Re-run workloads with: scripts/wsl/run.sh --offline ... for reproducible offline runs."
