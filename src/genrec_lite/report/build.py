"""Report generation (DESIGN.md §8.3)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

META_COLS = {"method", "slice", "n_samples", "seed"}


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


def aggregate_metrics_across_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed metrics into mean/std per method×slice."""
    if metrics.empty or "seed" not in metrics.columns:
        return metrics.copy()

    metric_cols = [c for c in metrics.columns if c not in META_COLS]
    group_cols = ["method", "slice"]
    if "n_samples" in metrics.columns:
        group_cols.append("n_samples")

    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby(["method", "slice"], dropna=False)
    for (method, slice_name), group in grouped:
        row: dict[str, Any] = {
            "method": method,
            "slice": slice_name,
            "n_samples": int(group["n_samples"].iloc[0]) if "n_samples" in group.columns else 0,
        }
        for col in metric_cols:
            if col == "n_samples":
                continue
            values = group[col].dropna().astype(float).tolist()
            if not values:
                row[col] = np.nan
                row[f"{col}_std"] = np.nan
            else:
                row[col] = float(np.mean(values))
                row[f"{col}_std"] = float(np.std(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_metrics_summary(run_dir: Path, summary: pd.DataFrame) -> None:
    """Save aggregated metrics summary as JSON."""
    records: list[dict[str, Any]] = []
    metric_cols = [c for c in summary.columns if c not in META_COLS and not c.endswith("_std")]
    for _, row in summary.iterrows():
        entry: dict[str, Any] = {
            "method": row["method"],
            "slice": row["slice"],
            "n_samples": int(row.get("n_samples", 0)),
            "metrics": {},
        }
        for col in metric_cols:
            std_col = f"{col}_std"
            entry["metrics"][col] = {
                "mean": float(row[col]) if pd.notna(row[col]) else None,
                "std": float(row[std_col])
                if std_col in summary.columns and pd.notna(row[std_col])
                else None,
            }
        records.append(entry)
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def build_results_markdown(
    metrics: pd.DataFrame,
    output_path: Path,
    primary_metric: str = "ndcg@20",
) -> None:
    """Build method × slice matrix markdown report."""
    if metrics.empty:
        output_path.write_text("# Results\n\nNo metrics available.\n", encoding="utf-8")
        return

    display = aggregate_metrics_across_seeds(metrics) if "seed" in metrics.columns else metrics
    methods = sorted(display["method"].unique())
    slices = sorted(display["slice"].unique())
    metric_cols = [c for c in display.columns if c not in META_COLS and not c.endswith("_std")]
    if primary_metric not in metric_cols and metric_cols:
        primary_metric = metric_cols[0]

    lines = ["# Results", "", f"Primary metric: `{primary_metric}`", ""]
    header = "| Method | " + " | ".join(slices) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(slices)) + " |"
    lines.extend([header, sep])

    for method in methods:
        row_vals: list[str] = [method]
        for slice_name in slices:
            match = display[(display["method"] == method) & (display["slice"] == slice_name)]
            if match.empty or primary_metric not in match.columns:
                row_vals.append("-")
            else:
                val = match.iloc[0][primary_metric]
                std_col = f"{primary_metric}_std"
                if pd.notna(val):
                    if std_col in match.columns and pd.notna(match.iloc[0][std_col]):
                        std = float(match.iloc[0][std_col])
                        row_vals.append(f"{val:.4f} (±{std:.4f})")
                    else:
                        row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append("-")
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append("")
    lines.append("## Detailed metrics")
    lines.append("")
    for _, row in display.iterrows():
        parts: list[str] = []
        for c in metric_cols:
            if pd.notna(row[c]):
                std_col = f"{c}_std"
                if std_col in display.columns and pd.notna(row[std_col]):
                    parts.append(f"{c}={row[c]:.4f} (±{row[std_col]:.4f})")
                else:
                    parts.append(f"{c}={row[c]:.4f}")
        lines.append(f"- **{row['method']}** / `{row['slice']}`: " + ", ".join(parts))
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
    summary_path = latest / "metrics_summary.json"
    metrics_path = latest / "metrics.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            data = json.load(f)
        rows: list[dict[str, Any]] = []
        for entry in data:
            row: dict[str, Any] = {
                "method": entry["method"],
                "slice": entry["slice"],
                "n_samples": entry.get("n_samples", 0),
            }
            for metric_name, stats in entry.get("metrics", {}).items():
                row[metric_name] = stats.get("mean")
                row[f"{metric_name}_std"] = stats.get("std")
            rows.append(row)
        return latest, pd.DataFrame(rows)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {latest}")
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return latest, pd.DataFrame(data)
