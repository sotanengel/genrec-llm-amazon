"""CLI entry point (DESIGN.md §11)."""

from __future__ import annotations

import logging
import random

import numpy as np
import typer
from rich.console import Console

from genrec_lite.config import find_project_root, load_data_config
from genrec_lite.data.loaders.amazon import prepare_amazon_dataset
from genrec_lite.data.stats import print_stats

app = typer.Typer(help="GenRec-lite: minimal GenRec reproduction CLI")
data_app = typer.Typer(help="Data preparation and statistics")
app.add_typer(data_app, name="data")

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


if __name__ == "__main__":
    app()
