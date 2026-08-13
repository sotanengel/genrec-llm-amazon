"""Report generation (DESIGN.md §8.3)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_run_dir(reports_root: Path, run_id: str | None = None) -> Path:
    """Create a timestamped run directory under reports/runs/."""
    if run_id is None:
        run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = reports_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    """Persist config, metrics, and git sha for a run."""
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    metrics.to_json(run_dir / "metrics.json", orient="records", indent=2)
    (run_dir / "git_sha.txt").write_text(_git_sha(), encoding="utf-8")


def build_results_markdown(
    metrics: pd.DataFrame,
    output_path: Path,
    primary_metric: str = "ndcg@20",
) -> None:
    """Build method × slice matrix markdown report."""
    if metrics.empty:
        output_path.write_text("# Results\n\nNo metrics available.\n", encoding="utf-8")
        return

    methods = sorted(metrics["method"].unique())
    slices = sorted(metrics["slice"].unique())
    metric_cols = [c for c in metrics.columns if c not in {"method", "slice", "n_samples"}]
    if primary_metric not in metric_cols and metric_cols:
        primary_metric = metric_cols[0]

    lines = ["# Results", "", f"Primary metric: `{primary_metric}`", ""]
    header = "| Method | " + " | ".join(slices) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(slices)) + " |"
    lines.extend([header, sep])

    for method in methods:
        row_vals: list[str] = [method]
        for slice_name in slices:
            match = metrics[(metrics["method"] == method) & (metrics["slice"] == slice_name)]
            if match.empty or primary_metric not in match.columns:
                row_vals.append("-")
            else:
                val = match.iloc[0][primary_metric]
                row_vals.append(f"{val:.4f}" if pd.notna(val) else "-")
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append("")
    lines.append("## Detailed metrics")
    lines.append("")
    for _, row in metrics.iterrows():
        lines.append(
            f"- **{row['method']}** / `{row['slice']}`: "
            + ", ".join(f"{c}={row[c]:.4f}" for c in metric_cols if pd.notna(row[c]))
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_latest_metrics(reports_root: Path) -> tuple[Path, pd.DataFrame]:
    """Load metrics from the most recent run directory."""
    runs_dir = reports_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs found under {runs_dir}")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {runs_dir}")
    latest = run_dirs[-1]
    metrics_path = latest / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {latest}")
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return latest, pd.DataFrame(data)
