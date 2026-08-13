"""CLI tests for benchmark scripts (issue #11)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_bench_prefill_dry_run_cpu_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_prefill.py",
            "--dry-run-cpu",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--json",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "model_id",
        "device",
        "batch_size",
        "tok_per_s_padded",
        "tok_per_s_real",
        "ms_per_batch_p50",
        "memory",
        "timestamp",
    ):
        assert key in payload


@pytest.mark.slow
def test_slow_marker_registered() -> None:
    """Placeholder slow test so pytest -m slow collects at least one test."""

    assert True
