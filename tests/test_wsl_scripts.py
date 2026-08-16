"""Static (RED) tests for scripts/wsl/* -- the WSL2 GPU bring-up script layer.

These tests run on Windows CI where WSL itself is not installed. They never
execute the scripts; they only check that each script is syntactically valid
and follows the conventions/regression-guards captured in DESIGN.md and the
WSL section of README.md (see scripts/wsl/env.sh for the incident writeups
these guard against).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WSL_DIR = Path(__file__).resolve().parent.parent / "scripts" / "wsl"
SH_SCRIPTS = sorted(WSL_DIR.glob("*.sh")) if WSL_DIR.exists() else []
PS1_SCRIPTS = sorted(WSL_DIR.glob("*.ps1")) if WSL_DIR.exists() else []
ALL_SCRIPTS = sorted(SH_SCRIPTS + PS1_SCRIPTS)

# Regression guard for the WSL2 + driver 610.47 incident: expandable_segments
# combined with VRAM oversubscription crashes CUDA with a misleading
# "device not ready" error. No script may ever (re-)export this variable.
_PYTORCH_ALLOC_CONF_EXPORT_RE = re.compile(r"export\s+PYTORCH_CUDA_ALLOC_CONF\b")


def test_wsl_dir_exists() -> None:
    assert WSL_DIR.exists(), f"expected {WSL_DIR} to exist"


def test_found_both_sh_and_ps1_scripts() -> None:
    assert SH_SCRIPTS, f"expected at least one scripts/wsl/*.sh under {WSL_DIR}"
    assert PS1_SCRIPTS, f"expected at least one scripts/wsl/*.ps1 under {WSL_DIR}"


@pytest.mark.parametrize("script", SH_SCRIPTS, ids=lambda p: p.name)
def test_sh_script_parses_with_bash_n(script: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this system")
    result = subprocess.run(
        [bash, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed for {script}:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("script", SH_SCRIPTS, ids=lambda p: p.name)
def test_sh_script_has_set_euo_pipefail(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text, f"{script} is missing 'set -euo pipefail'"


@pytest.mark.parametrize("script", SH_SCRIPTS, ids=lambda p: p.name)
def test_sh_script_uses_lf_line_endings(script: Path) -> None:
    raw = script.read_bytes()
    assert b"\r\n" not in raw, f"{script} has CRLF line endings -- breaks the shebang under WSL"


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_script_never_exports_pytorch_cuda_alloc_conf(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert not _PYTORCH_ALLOC_CONF_EXPORT_RE.search(text), (
        f"{script} must never export PYTORCH_CUDA_ALLOC_CONF -- expandable_segments "
        "combined with VRAM oversubscription is known to crash CUDA on WSL2 with "
        "driver 610.47 (RuntimeError: CUDA driver error: device not ready). "
        "See scripts/wsl/env.sh."
    )


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_script_has_no_literal_windows_drive_path(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "C:\\" not in text, (
        f"{script} contains a literal 'C:\\' path -- pass /mnt/c/... (or a pure Linux "
        "path) to WSL instead; backslash-prefixed Windows paths are misinterpreted as "
        "escape sequences inside WSL."
    )


@pytest.mark.parametrize("script", PS1_SCRIPTS, ids=lambda p: p.name)
def test_ps1_script_has_no_and_or_or_operators(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "&&" not in text, f"{script} contains '&&', which is a parser error in PowerShell 5.1"
    assert "||" not in text, f"{script} contains '||', which is a parser error in PowerShell 5.1"


def test_resume_systemd_service_is_restartable_and_logged() -> None:
    template = WSL_DIR / "systemd" / "genrec-m3-resume.service.in"
    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "Restart=on-failure" in text
    assert "RestartSec=" in text
    assert "resume_m3_train_head.sh" in text
    assert "StandardOutput=journal" in text
    assert "StandardError=journal" in text


def test_service_manager_supports_lifecycle_and_logs() -> None:
    manager = WSL_DIR / "manage_m3_resume_service.sh"
    assert manager.exists()
    text = manager.read_text(encoding="utf-8")
    for command in ("install", "start", "status", "logs"):
        assert command in text
    assert "systemctl --user daemon-reload" in text
    assert "systemctl --user enable" in text
    assert "loginctl enable-linger" in text
    assert "journalctl --user-unit" in text
    assert "wsl --shutdown" not in text.lower()


def test_resume_pipeline_records_durable_completion_marker() -> None:
    resume = (WSL_DIR / "resume_m3_train_head.sh").read_text(encoding="utf-8")
    assert "m3_resume.complete" in resume
    assert "mv " in resume


def test_monitor_handles_systemd_resume_and_completion() -> None:
    monitor = (WSL_DIR / "monitor_m3_production.sh").read_text(encoding="utf-8")
    assert "systemctl --user is-active" in monitor
    assert "systemctl --user is-failed" in monitor
    assert "m3_resume.complete" in monitor
    assert "phase=failed" in monitor
