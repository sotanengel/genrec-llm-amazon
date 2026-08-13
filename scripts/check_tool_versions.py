"""Verify the ruff version is pinned identically in every place it appears.

This is the structural fix for the ruff-version-drift class of bug that has
already been hand-fixed twice in this repo (see issue #13, P1): the
`.pre-commit-config.yaml` `rev:` for the ruff-pre-commit repo must match the
`ruff==` pin in `pyproject.toml`'s `[dependency-groups].lint`, exactly, so
that `uv run ruff` and the pre-commit ruff hook can never silently diverge.

Network-free: only reads local files. Usable both as a library (imported by
tests/test_toolchain_consistency.py) and as a standalone CLI / pre-commit
hook (`python scripts/check_tool_versions.py`).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_RUFF_REPO_MARKER = "astral-sh/ruff-pre-commit"
_REV_RE = re.compile(r"^\s*rev:\s*['\"]?([^'\"#\s]+)['\"]?\s*$")


def normalize_version(version: str) -> str:
    """Strip an optional leading 'v' so 'v0.16.2' == '0.16.2'."""
    return version[1:] if version.startswith("v") else version


def get_precommit_ruff_rev(path: Path = PRE_COMMIT_CONFIG) -> str:
    """Parse the `rev:` pinned for the ruff-pre-commit repo.

    Deliberately avoids a YAML dependency here: a tiny line-oriented scan is
    enough for this repo's flat, single-document config, and keeps this
    script runnable with only the stdlib.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if _RUFF_REPO_MARKER in line:
            for follow in lines[i + 1 : i + 5]:
                match = _REV_RE.match(follow)
                if match:
                    return normalize_version(match.group(1))
            break
    raise ValueError(f"Could not find a `rev:` pinned for {_RUFF_REPO_MARKER!r} in {path}")


def get_pyproject_ruff_pin(path: Path = PYPROJECT) -> str:
    """Parse the exact `ruff==X.Y.Z` pin out of `[dependency-groups].lint`."""
    with path.open("rb") as f:
        data = tomllib.load(f)

    groups = data.get("dependency-groups", {})
    lint_group = groups.get("lint", [])
    for entry in lint_group:
        if not isinstance(entry, str):
            continue
        match = re.match(r"^ruff==([^\s;]+)$", entry.strip())
        if match:
            return normalize_version(match.group(1))

    raise ValueError(
        f"Could not find an exact `ruff==X.Y.Z` pin in "
        f"[dependency-groups].lint of {path}. Found entries: {lint_group!r}"
    )


def check() -> tuple[bool, str]:
    """Return (ok, message) describing whether the two pins agree."""
    precommit_rev = get_precommit_ruff_rev()
    pyproject_pin = get_pyproject_ruff_pin()
    if precommit_rev == pyproject_pin:
        return True, f"OK: ruff pinned at {pyproject_pin!r} in both places."
    return False, (
        "MISMATCH: .pre-commit-config.yaml pins ruff rev "
        f"{precommit_rev!r} but pyproject.toml's [dependency-groups].lint "
        f"pins ruff=={pyproject_pin!r}. These must be identical so `uv run "
        "ruff` and the pre-commit ruff hook cannot silently diverge "
        "(see issue #13, P1)."
    )


def main() -> int:
    ok, message = check()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
