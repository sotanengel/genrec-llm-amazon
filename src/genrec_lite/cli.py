"""CLI entry point (DESIGN.md §11)."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import typer
from rich.console import Console

from genrec_lite.config import (
    find_project_root,
    load_data_config,
    load_exp_config,
    load_llm_config,
    load_verbalizer_config,
)
from genrec_lite.data.loaders.amazon import prepare_amazon_dataset
from genrec_lite.data.schema import read_parquet_bundle
from genrec_lite.data.stats import print_stats
from genrec_lite.encode.cache import CacheKeyConfig, HiddenStateCache, compute_cache_key
from genrec_lite.encode.prefill import PrefillEncoder
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.baselines import build_baseline
from genrec_lite.report.build import (
    aggregate_metrics_across_seeds,
    build_results_markdown,
    create_run_dir,
    load_latest_metrics,
    save_metrics_summary,
    save_run_metadata,
)
from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.templates import build_verbalizer

app = typer.Typer(help="GenRec-lite: minimal GenRec reproduction CLI")
data_app = typer.Typer(help="Data preparation and statistics")
eval_app = typer.Typer(help="Evaluation commands")
report_app = typer.Typer(help="Report generation")
verbalize_app = typer.Typer(help="Verbalizer commands")
encode_app = typer.Typer(help="Encoding commands")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")
app.add_typer(verbalize_app, name="verbalize")
app.add_typer(encode_app, name="encode")

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
    seeds_to_run = [run_seed] if seed is not None else list(exp_config.seeds)

    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(code=1)

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    eval_samples = samples.filter(pl.col("split") == exp_config.eval_split)

    all_results: list[pd.DataFrame] = []
    for current_seed in seeds_to_run:
        _set_seed(current_seed)
        for baseline_name in exp_config.baselines:
            console.print(f"[bold]Evaluating[/bold] {baseline_name} (seed={current_seed})")
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
            result["seed"] = current_seed
            all_results.append(result)

    metrics = pd.concat(all_results, ignore_index=True)
    summary = aggregate_metrics_across_seeds(metrics)
    reports_root = root / "reports"
    run_dir = create_run_dir(reports_root)
    config_dict: dict[str, Any] = exp_config.model_dump()
    config_dict["seeds"] = seeds_to_run
    save_run_metadata(run_dir, config_dict, metrics)
    save_metrics_summary(run_dir, summary)
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


def _sample_from_row(row: dict[str, Any]) -> Sample:
    return Sample(
        user_id=int(row["user_id"]),
        cutoff_ts=int(row["cutoff_ts"]),
        target_item=int(row["target_item"]),
        history=[int(h) for h in row["history"]],
        is_repeat=bool(row["is_repeat"]),
        target_is_cold=bool(row["target_is_cold"]),
    )


@verbalize_app.command("render")
def verbalize_render(
    dataset: str = typer.Option(..., "--dataset", help="Dataset config name"),
    verbalizer: str = typer.Option("v1_full", "--verbalizer", help="Verbalizer config name"),
    n: int = typer.Option(20, "--n", help="Number of samples to dump"),
    output: str = typer.Option("reports/verbalizer_samples.md", "--output"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render verbalizer samples to markdown for human review."""
    _setup_logging(verbose)
    root = find_project_root()
    data_config = load_data_config(dataset, config_dir=root / "configs")
    verb_config = load_verbalizer_config(verbalizer, config_dir=root / "configs")
    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(code=1)

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    renderer = build_verbalizer(verbalizer)
    budget = TokenBudget(
        max_tokens=verb_config.max_tokens, tokenizer_name=verb_config.tokenizer_name
    )
    subset = samples.head(n)

    lines = [f"# Verbalizer samples ({verbalizer})", ""]
    for i, row in enumerate(subset.iter_rows(named=True), start=1):
        sample = _sample_from_row(row)
        prompt = renderer.render(sample, items, users, interactions, budget)
        lines.extend([f"## Sample {i}", "", "```", prompt, "```", ""])

    output_path = root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {output_path}")


@encode_app.command("run")
def encode_run(
    dataset: str = typer.Option(..., "--dataset", help="Dataset config name"),
    model: str = typer.Option("qwen3-1.7b-base", "--model", help="LLM config name"),
    verbalizer: str = typer.Option("v1_full", "--verbalizer", help="Verbalizer config name"),
    cache_dir: str = typer.Option("cache/hidden_states", "--cache-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode samples with frozen LLM prefill and cache hidden states."""
    _setup_logging(verbose)
    root = find_project_root()
    data_config = load_data_config(dataset, config_dir=root / "configs")
    llm_config = load_llm_config(model, config_dir=root / "configs")
    verb_config = load_verbalizer_config(verbalizer, config_dir=root / "configs")
    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(code=1)

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    renderer = build_verbalizer(verbalizer)
    budget = TokenBudget(
        max_tokens=verb_config.max_tokens, tokenizer_name=verb_config.tokenizer_name
    )
    texts: list[str] = []
    sample_ids: list[int] = []
    for row in samples.iter_rows(named=True):
        sample = _sample_from_row(row)
        texts.append(renderer.render(sample, items, users, interactions, budget))
        sample_ids.append(int(row["sample_id"]))

    encoder = PrefillEncoder(
        model_id=llm_config.model_id,
        dtype=llm_config.dtype,
        pooling=llm_config.pooling,
        max_len=llm_config.max_len,
        quantize=llm_config.quantize,
    )
    hidden_dim = int(encoder.encode_batch([texts[0]]).shape[1])
    cache_key = compute_cache_key(
        CacheKeyConfig(
            model_id=llm_config.model_id,
            verbalizer_name=verbalizer,
            verbalizer_config=verb_config.model_dump(),
            max_len=llm_config.max_len,
        )
    )
    cache = HiddenStateCache(root / cache_dir, cache_key, len(texts), hidden_dim)
    if cache.exists():
        console.print(f"[yellow]Cache hit:[/yellow] {cache.memmap_path}")
        raise typer.Exit(code=0)

    batch_size = 8
    chunks: list[Any] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        chunks.append(encoder.encode_batch(batch))

    hidden = torch.cat(chunks, dim=0)
    cache.save(sample_ids, hidden)
    msg = (
        f"[green]Cached[/green] {len(texts)} vectors "
        f"({cache.expected_bytes} bytes) -> {cache.memmap_path}"
    )
    console.print(msg)


if __name__ == "__main__":
    app()
