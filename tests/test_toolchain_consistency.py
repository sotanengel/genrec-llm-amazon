"""Toolchain-pin consistency checks (issue #13, P1).

ruff's version has drifted between `.pre-commit-config.yaml` and
`pyproject.toml` and been hand-fixed twice already (commits `b8eb3e5`,
`d3809bf`). These tests are the structural fix: they fail loudly, offline,
whenever the two pins disagree again, instead of relying on someone noticing
a red CI run caused by a brand-new PyPI release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_tool_versions  # noqa: E402


def test_normalize_version_strips_leading_v() -> None:
    assert check_tool_versions.normalize_version("v0.16.2") == "0.16.2"
    assert check_tool_versions.normalize_version("0.16.2") == "0.16.2"


def test_precommit_config_pins_a_ruff_rev() -> None:
    rev = check_tool_versions.get_precommit_ruff_rev()
    assert rev, "expected a non-empty ruff rev in .pre-commit-config.yaml"


def test_pyproject_pins_an_exact_ruff_version() -> None:
    pin = check_tool_versions.get_pyproject_ruff_pin()
    assert pin, "expected an exact ruff==X.Y.Z pin in [dependency-groups].lint"


def test_ruff_version_matches_between_precommit_and_pyproject() -> None:
    """The one test that would have caught commits b8eb3e5 and d3809bf."""
    ok, message = check_tool_versions.check()
    assert ok, message


@pytest.mark.parametrize(
    "path", [check_tool_versions.PRE_COMMIT_CONFIG, check_tool_versions.PYPROJECT]
)
def test_referenced_files_exist(path: Path) -> None:
    assert path.is_file(), f"expected {path} to exist"
