"""CLI entry point (DESIGN.md §11)."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import typer
from rich.console import Console

from genrec_lite.config import find_project_root, load_data_config, load_exp_config
from genrec_lite.data.loaders.amazon import prepare_amazon_dataset
from genrec_lite.data.schema import read_parquet_bundle
from genrec_lite.data.stats import print_stats
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.baselines import build_baseline
from genrec_lite.report.build import (
    build_results_markdown,
    create_run_dir,
    load_latest_metrics,
    save_run_metadata,
)

app = typer.Typer(help="GenRec-lite: minimal GenRec reproduction CLI")
data_app = typer.Typer(help="Data preparation and statistics")
eval_app = typer.Typer(help="Evaluation commands")
report_app = typer.Typer(help="Report generation")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


@data_app.command("prepare")
def data_prepare(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name (config file stem)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download and prepare dataset parquet bundle."""
    _setup_logging(verbose)
    root = find_project_root()
    config = load_data_config(dataset, config_dir=root / "configs")
    _set_seed(config.seed)

    output_dir = root / config.output_dir
    try:
        console.print(f"[bold]Preparing dataset:[/bold] {dataset} (category={config.category})")
        prepare_amazon_dataset(
            category=config.category,
            output_dir=output_dir,
            split_strategy=config.split_strategy,
            cold_threshold=config.cold_threshold,
        )
        console.print(f"[green]Done.[/green] Output: {output_dir}")
    except FileNotFoundError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Data error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[red]Download error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@data_app.command("stats")
def data_stats(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name (config file stem)"),
) -> None:
    """Print dataset statistics from processed parquet."""
    root = find_project_root()
    config = load_data_config(dataset, config_dir=root / "configs")
    data_dir = root / config.output_dir

    if not data_dir.exists():
        console.print(
            f"[red]Data directory not found:[/red] {data_dir}\n"
            "Run `python -m genrec_lite data prepare --dataset {dataset}` first."
        )
        raise typer.Exit(code=1)

    try:
        print_stats(data_dir, console=console)
    except Exception as exc:
        console.print(f"[red]Failed to compute stats:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@eval_app.command("run")
def eval_run(
    exp: str = typer.Option(..., "--exp", help="Experiment config name"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run baseline evaluation for an experiment config."""
    _setup_logging(verbose)
    root = find_project_root()
    exp_config = load_exp_config(exp, config_dir=root / "configs")
    data_config = load_data_config(exp_config.dataset, config_dir=root / "configs")
    run_seed = seed if seed is not None else exp_config.seeds[0]
    _set_seed(run_seed)

    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(code=1)

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    eval_samples = samples.filter(pl.col("split") == exp_config.eval_split)

    all_results: list[pd.DataFrame] = []
    for baseline_name in exp_config.baselines:
        console.print(f"[bold]Evaluating[/bold] {baseline_name}")
        model = build_baseline(baseline_name, sasrec_config=exp_config.sasrec)
        model.fit(interactions, items)
        result = evaluate(
            score_fn=model.score_batch,
            samples=eval_samples,
            items=items,
            interactions=interactions,
            ks=tuple(exp_config.ks),
            cold_threshold=exp_config.cold_threshold,
            method=baseline_name,
        )
        all_results.append(result)

    metrics = pd.concat(all_results, ignore_index=True)
    reports_root = root / "reports"
    run_dir = create_run_dir(reports_root)
    config_dict: dict[str, Any] = exp_config.model_dump()
    config_dict["seed"] = run_seed
    save_run_metadata(run_dir, config_dict, metrics)
    build_results_markdown(metrics, reports_root / "results.md")
    console.print(f"[green]Done.[/green] Run saved to {run_dir}")


@report_app.command("build")
def report_build(
    exp: str = typer.Option(..., "--exp", help="Experiment config name (for logging)"),
) -> None:
    """Build results.md from the latest evaluation run."""
    root = find_project_root()
    _ = load_exp_config(exp, config_dir=root / "configs")
    reports_root = root / "reports"
    try:
        run_dir, metrics = load_latest_metrics(reports_root)
    except FileNotFoundError as exc:
        console.print(f"[red]Report error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    output_path = reports_root / "results.md"
    build_results_markdown(metrics, output_path)
    console.print(f"[green]Report written:[/green] {output_path} (from {run_dir.name})")


if __name__ == "__main__":
    app()
