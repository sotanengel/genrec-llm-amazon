#!/usr/bin/env python3
"""Cross-platform task runner shared by the Makefile and pre-commit.

`make` is not available on Windows, where this repo is authored, but the
pre-commit hooks still have to run there. Defining each command once here and
having both the Makefile and `.pre-commit-config.yaml` delegate to it keeps a
single source of truth without depending on a shell or on GNU make.

It also lets a task set environment variables (``test-precommit`` needs
``GENREC_NO_NETWORK=1``), which pre-commit's ``language: system`` cannot express
because it splits ``entry`` itself instead of going through a shell.

Run inside the project environment, e.g.::

    uv run --frozen python scripts/dev.py typecheck
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field

PYTEST_TIMEOUT = "60"
SLOW_TIMEOUT = "600"

# Ruff and mypy deliberately cover the same file set in every entry point so
# that CI, the Makefile and pre-commit can never disagree about scope -- the
# drift that produced commit d3809bf (scripts/ linted by pre-commit but not CI).
CHECKED_PATHS = ["src", "tests", "scripts"]

# mypy --strict covers src and scripts only: tests/ currently has ~100
# pre-existing errors (see issue #13). Widening this is a separate unit.
TYPED_PATHS = ["src", "scripts"]


@dataclass(frozen=True)
class Task:
    """A single dev command: argv plus any environment overrides."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # pytest exits 5 when it collects nothing. For `test-slow` that is the
    # expected state until some test carries the marker, so tolerate just that
    # code rather than failing the target.
    ok_codes: tuple[int, ...] = (0,)


TASKS: dict[str, Task] = {
    "lint": Task(["ruff", "check", *CHECKED_PATHS]),
    "format": Task(["ruff", "format", *CHECKED_PATHS]),
    "format-check": Task(["ruff", "format", "--check", *CHECKED_PATHS]),
    "typecheck": Task(
        [
            "mypy",
            "--config-file=pyproject.toml",
            "--strict",
            "--explicit-package-bases",
            *TYPED_PATHS,
        ]
    ),
    "check-model-configs": Task(
        [sys.executable, "scripts/check_model_configs.py", "configs/model/llm"]
    ),
    "test-fast": Task(
        ["pytest", "-x", "-n", "auto", "--timeout", PYTEST_TIMEOUT, "tests/", "-m", "not slow"]
    ),
    "test-cpu": Task(
        [
            "pytest",
            "-n",
            "auto",
            "--timeout",
            PYTEST_TIMEOUT,
            "tests/",
            "-m",
            "not slow and not gpu",
        ]
    ),
    # Offline-safe by construction: network tests are auto-skipped rather than
    # attempted, so a commit works without a network connection (issue #13, P4).
    "test-precommit": Task(
        ["pytest", "-x", "-n", "auto", "--timeout", PYTEST_TIMEOUT, "tests/", "-m", "not slow"],
        env={"GENREC_NO_NETWORK": "1"},
    ),
    "test-slow": Task(
        ["pytest", "tests/", "-m", "slow", "--timeout", SLOW_TIMEOUT],
        ok_codes=(0, 5),
    ),
    # GPU tests never run on GitHub-hosted runners (no CUDA device); this is the
    # local/WSL gate.
    "test-gpu": Task(["pytest", "tests/", "-m", "gpu", "--timeout", SLOW_TIMEOUT]),
}


def run(name: str) -> int:
    task = TASKS[name]
    env = {**os.environ, **task.env}
    try:
        completed = subprocess.run(task.argv, env=env, check=False)  # noqa: S603
    except FileNotFoundError:
        print(
            f"dev.py: '{task.argv[0]}' not found. Run inside the project environment, "
            f"e.g. `uv run --frozen python scripts/dev.py {name}`.",
            file=sys.stderr,
        )
        return 127
    if completed.returncode in task.ok_codes:
        if completed.returncode != 0:
            print(f"dev.py: '{name}' exited {completed.returncode}, tolerated by design.")
        return 0
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASKS), help="Task to run")
    args = parser.parse_args(argv)
    task_name: str = args.task
    return run(task_name)


if __name__ == "__main__":
    raise SystemExit(main())
