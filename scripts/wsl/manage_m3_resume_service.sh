#!/usr/bin/env bash
# Install and manage the durable M3 train-head/report user service.
# Usage: bash scripts/wsl/manage_m3_resume_service.sh install|start|status|logs|restart|stop|uninstall
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICE_NAME="genrec-m3-resume.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${UNIT_DIR}/${SERVICE_NAME}"
TEMPLATE="${ROOT}/scripts/wsl/systemd/${SERVICE_NAME}.in"

require_user_systemd() {
  if ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "systemd user manager is unavailable. Enable systemd in /etc/wsl.conf and restart the distro." >&2
    return 1
  fi
}

enable_linger() {
  local linger
  linger="$(loginctl show-user "${USER}" --property=Linger --value 2>/dev/null || true)"
  if [[ "${linger}" != "yes" ]]; then
    echo "Enabling systemd linger so the resume service survives terminal logout."
    if ! sudo loginctl enable-linger "${USER}"; then
      echo "Failed to enable linger; run: sudo loginctl enable-linger ${USER}" >&2
      return 1
    fi
  fi
}

install_service() {
  local escaped_root tmp
  require_user_systemd
  enable_linger
  mkdir -p "${UNIT_DIR}" "${ROOT}/logs"
  escaped_root="${ROOT//\\/\\\\}"
  escaped_root="${escaped_root//&/\\&}"
  escaped_root="${escaped_root//|/\\|}"
  tmp="${UNIT_PATH}.tmp"
  sed "s|@PROJECT_ROOT@|${escaped_root}|g" "${TEMPLATE}" >"${tmp}"
  mv "${tmp}" "${UNIT_PATH}"
  systemctl --user daemon-reload
  systemctl --user enable "${SERVICE_NAME}"
  echo "installed ${UNIT_PATH}"
}

case "${1:-}" in
  install)
    install_service
    ;;
  start)
    require_user_systemd
    systemctl --user start "${SERVICE_NAME}"
    systemctl --user --no-pager status "${SERVICE_NAME}"
    ;;
  restart)
    require_user_systemd
    systemctl --user restart "${SERVICE_NAME}"
    systemctl --user --no-pager status "${SERVICE_NAME}"
    ;;
  status)
    require_user_systemd
    systemctl --user --no-pager status "${SERVICE_NAME}" || true
    bash "${ROOT}/scripts/wsl/monitor_m3_production.sh" --once
    ;;
  logs)
    require_user_systemd
    exec journalctl --user-unit "${SERVICE_NAME}" -f
    ;;
  stop)
    require_user_systemd
    systemctl --user stop "${SERVICE_NAME}"
    ;;
  uninstall)
    require_user_systemd
    systemctl --user disable --now "${SERVICE_NAME}" || true
    rm -f "${UNIT_PATH}"
    systemctl --user daemon-reload
    ;;
  *)
    echo "usage: $0 install|start|status|logs|restart|stop|uninstall" >&2
    exit 2
    ;;
esac
