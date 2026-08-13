"""CLI tests for benchmark scripts (issue #11).

The tests below (``test_build_filler_text_reaches_requested_length``,
``test_tok_per_s_padded_matches_real_for_identical_inputs``,
``test_bench_prefill_json_schema_has_actual_seq_len_fields``) are regression
tests for a measurement-correctness bug: ``bench_prefill.py`` used to build
filler text from a fixed characters-per-token guess
(``"benchmark " * (max_len // 8)``), which undershot real tokenizers by
roughly 4-8x, and computed ``tok_per_s_padded`` from the *nominal*
``--seq-len`` instead of the tokenizer's real padded tensor width, inflating
it by the same factor. Both are exercised here against the real tiny-gpt2
tokenizer so a reintroduced fixed-ratio assumption fails loudly instead of
silently producing wrong numbers again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bench_prefill  # noqa: E402


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


def test_bench_prefill_json_schema_has_actual_seq_len_fields(tmp_path: Path) -> None:
    """New fields that make the achieved (real) measurement visible.

    ``seq_len`` alone is what caused the original bug to go unnoticed: it
    reported the *requested* length even when the actually-executed sequence
    was far shorter. ``seq_len_actual`` and ``padded_width`` must both be
    present so a reader can tell what was really run.
    """
    out = tmp_path / "bench.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_prefill.py",
            "--dry-run-cpu",
            "--seq-len",
            "150",
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
    for key in ("seq_len", "seq_len_actual", "padded_width", "warning"):
        assert key in payload
    # The requested length is reachable well within tiny-gpt2's 1024-token
    # context, so the achieved length should match it exactly (filler text is
    # grown until truncation at max_length=seq_len yields seq_len tokens).
    assert payload["seq_len_actual"] == payload["seq_len"]
    assert payload["warning"] is None


def test_build_filler_text_reaches_requested_length(tiny_model_id: str) -> None:
    """Regression test for defect 1 (undershooting filler text).

    The old implementation guessed ``"benchmark " * (max_len // 8)`` -- a
    fixed characters-per-token ratio that does not hold for real tokenizers
    (it undershot by roughly 4-8x for the models this project uses). This
    builds filler text for the real tiny-gpt2 tokenizer and asserts the
    post-truncation length is exactly the requested target, with no ratio
    assumed anywhere in the process.
    """
    from genrec_lite.encode.prefill import PrefillEncoder

    seq_len = 200
    encoder = PrefillEncoder(model_id=tiny_model_id, dtype="float32", max_len=seq_len, device="cpu")
    text = bench_prefill.build_filler_text(encoder, seq_len)
    actual = encoder.token_lengths([text])[0]
    assert actual == seq_len


def test_tok_per_s_padded_matches_real_for_identical_inputs(tiny_model_id: str) -> None:
    """Regression test for defect 2 (padded throughput used the nominal length).

    With identical strings in every batch slot and ``padding="longest"``,
    there is no actual padding, so the real padded tensor width equals the
    real (post-truncation) sequence length and ``tok_per_s_padded`` should
    closely track ``tok_per_s_real``. The old implementation computed
    ``padded_tokens`` from the nominal ``--seq-len`` regardless of the real
    tensor width, inflating ``tok_per_s_padded`` roughly 8x for this
    benchmark's default configuration.
    """
    del tiny_model_id  # only needed to auto-mark this test `network`
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_prefill.py",
            "--dry-run-cpu",
            "--seq-len",
            "150",
            "--batch-size",
            "3",
            "--padding",
            "longest",
            "--steps",
            "2",
            "--warmup",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    real = payload["tok_per_s_real"]
    padded = payload["tok_per_s_padded"]
    assert real > 0
    assert padded == pytest.approx(real, rel=0.05)


@pytest.mark.slow
def test_slow_marker_registered() -> None:
    """Placeholder slow test so pytest -m slow collects at least one test."""

    assert True
