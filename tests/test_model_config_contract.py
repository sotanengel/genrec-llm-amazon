"""Contract tests for configs/model/llm/*.yaml (DESIGN.md §2.4.4, issue #12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genrec_lite.config import validate_llm_config_file

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "model" / "llm"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_model_config_satisfies_contract(path: Path) -> None:
    """Every model YAML must pin revision and declare license/commercial_use_ok."""
    validate_llm_config_file(path)
