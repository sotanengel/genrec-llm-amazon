#!/usr/bin/env bash
set -euo pipefail

# scripts/wsl/bootstrap.sh
#
# Idempotent WSL2 (Ubuntu-24.04) provisioning for this repo: apt packages, uv,
# Python 3.12, clone-or-fetch the repo into ~/src, `uv sync`, and install
# ~/.config/genrec/env.sh. Safe to re-run at any time.
#
# Usage (from inside WSL, after scripts/wsl/install_distro.ps1 has provisioned
# the distro and user):
#   bash scripts/wsl/bootstrap.sh
#
# Or from Windows via scripts/wsl/Invoke-Wsl.ps1 -Command "scripts/wsl/bootstrap.sh"
# (first run: point Invoke-Wsl.ps1's -Command at the /mnt/c copy of this script,
# since the ~/src clone does not exist yet).
#
# Environment overrides:
#   GENREC_REPO_URL         git remote to clone (default: GitHub SSH remote)
#   GENREC_REPO_DIR         target clone dir (default: $HOME/src/genrec-llm-amazon)
#   GENREC_WIN_WORKTREE     Windows-side worktree to add as a 'win' remote, if present

REPO_URL="${GENREC_REPO_URL:-git@github.com:sotanengel/genrec-llm-amazon.git}"
REPO_DIR="${GENREC_REPO_DIR:-${HOME}/src/genrec-llm-amazon}"

log() { printf '[bootstrap] %s\n' "$*" >&2; }
fail() { printf '[bootstrap] FATAL: %s\n' "$*" >&2; exit 1; }

log "Step 1/6: apt packages (build-essential git curl ca-certificates pkg-config)"
if ! command -v sudo >/dev/null 2>&1; then
  fail "sudo not found. Run scripts/wsl/install_distro.ps1 first to provision the WSL user."
fi
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates pkg-config

log "Step 2/6: uv (standalone installer, no sudo)"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  fail "'uv' still not on PATH after installing to ~/.local/bin. Check the installer output above."
fi
log "uv version: $(uv --version)"

log "Step 3/6: Python 3.12 via uv"
uv python install 3.12

log "Step 4/6: clone-or-fetch repo into ${REPO_DIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  log "repo already present, fetching..."
  git -C "${REPO_DIR}" fetch --all --prune
else
  log "cloning ${REPO_URL}"
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

# Optional secondary remote pointing at the Windows-side worktree, so unpushed
# branches can transfer without a GitHub round-trip. GitHub remains the source of
# truth; this is a convenience remote only. Best-effort -- never fatal.
WIN_WORKTREE="${GENREC_WIN_WORKTREE:-/mnt/c/Users/na-g-/genrec-llm-amazon}"
if [ -d "${WIN_WORKTREE}" ]; then
  if ! git -C "${REPO_DIR}" remote get-url win >/dev/null 2>&1; then
    if git -C "${REPO_DIR}" remote add win "${WIN_WORKTREE}"; then
      log "added secondary remote 'win' -> ${WIN_WORKTREE}"
    fi
  fi
fi

log "Step 5/6: uv sync"
cd "${REPO_DIR}"
SYNC_OK=0
if [ -f uv.lock ]; then
  if uv sync --frozen --extra gpu --group dev; then
    SYNC_OK=1
  else
    log "'uv sync --frozen --extra gpu --group dev' failed; trying unfrozen fallbacks."
  fi
else
  log "NOTE: uv.lock not found yet in this checkout. A separate work unit is"
  log "      introducing [dependency-groups] + a committed uv.lock; until that"
  log "      lands, falling back to an unfrozen 'uv sync' with best-effort extras."
fi
if [ "${SYNC_OK}" -eq 0 ]; then
  for args in "--extra gpu --group dev" "--group dev" "--extra gpu --extra dev" "--extra dev" ""; do
    log "trying: uv sync ${args}"
    # shellcheck disable=SC2086
    if uv sync ${args}; then
      SYNC_OK=1
      log "uv sync succeeded with args: '${args:-<none>}'"
      break
    fi
  done
fi
if [ "${SYNC_OK}" -eq 0 ]; then
  fail "uv sync failed with every extras/group combination tried. Inspect" \
       "pyproject.toml's [project.optional-dependencies] / [dependency-groups]" \
       "and run 'uv sync' by hand to see the underlying resolver error."
fi

log "Step 6/6: install ~/.config/genrec/env.sh"
mkdir -p "${HOME}/.config/genrec"
chmod 700 "${HOME}/.config/genrec"
cp "${REPO_DIR}/scripts/wsl/env.sh" "${HOME}/.config/genrec/env.sh"

log "Bootstrap complete. Repo at ${REPO_DIR}."
log "Next steps:"
log "  - (optional, gated models only) write your token to ~/.config/genrec/hf_token and chmod 600 it"
log "  - bash scripts/wsl/doctor.sh          # verify the whole GPU stack"
log "  - bash scripts/wsl/fetch_models.sh    # download models declared in configs/model/llm/*.yaml"
