"""CLI entry point (DESIGN.md §11)."""

from __future__ import annotations

import logging
import os
import random
import subprocess
from datetime import UTC, datetime
from typing import Annotated, Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import typer
from rich.console import Console

from genrec_lite.config import (
    LLMConfig,
    VerbalizerYamlConfig,
    find_project_root,
    load_data_config,
    load_exp_config,
    load_llm_config,
    load_m3_exp_config,
    load_verbalizer_config,
)
from genrec_lite.data.loaders.amazon import prepare_amazon_dataset
from genrec_lite.data.schema import read_parquet_bundle, read_train_samples
from genrec_lite.data.stats import print_stats
from genrec_lite.encode.batching import batch_indices_by_token_budget
from genrec_lite.encode.cache import CacheKeyConfig, CacheScope, HiddenStateCache, compute_cache_key
from genrec_lite.encode.prefill import ENCODER_VERSION, PrefillEncoder
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.baselines import build_baseline
from genrec_lite.models.genrec_lite import GenRecLite
from genrec_lite.report.build import (
    aggregate_metrics_across_seeds,
    build_results_markdown,
    create_run_dir,
    load_latest_metrics,
    save_metrics_summary,
    save_run_metadata,
)
from genrec_lite.train.head_trainer import HeadTrainer, build_log_q
from genrec_lite.train.hidden_store import HiddenStateStore
from genrec_lite.train.item_init import apply_item_init_freeze, build_item_init_matrix
from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.budget import get_tokenizer
from genrec_lite.verbalize.templates import TemplateVerbalizer, build_verbalizer_from_config

app = typer.Typer(help="GenRec-lite: minimal GenRec reproduction CLI")
data_app = typer.Typer(help="Data preparation and statistics")
eval_app = typer.Typer(help="Evaluation commands")
report_app = typer.Typer(help="Report generation")
verbalize_app = typer.Typer(help="Verbalizer commands")
encode_app = typer.Typer(help="Encoding commands")
train_app = typer.Typer(help="Training commands")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")
app.add_typer(verbalize_app, name="verbalize")
app.add_typer(encode_app, name="encode")
app.add_typer(train_app, name="train")

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


def _encode_deterministic(llm_config: LLMConfig) -> bool:
    return llm_config.deterministic or os.environ.get("GENREC_DETERMINISTIC") == "1"


def _resolve_tokenizer_name(
    verb_config: VerbalizerYamlConfig,
    encoder_model_id: str | None = None,
) -> str:
    if verb_config.tokenizer_name is not None:
        return verb_config.tokenizer_name
    if encoder_model_id is not None:
        return encoder_model_id
    return "gpt2"


def _resolve_tokenizer_revision(
    verb_config: VerbalizerYamlConfig,
    encoder_revision: str | None = None,
) -> str | None:
    """Resolve the revision to pin the budget tokenizer to, mirroring
    `_resolve_tokenizer_name`'s null-means-encoder's-tokenizer semantics.

    `tokenizer_name: null` means "use the encoder's own tokenizer", so it must
    also be pinned to the *encoder's* revision (DESIGN.md §2.4.4) -- otherwise
    a revision-pinned download (see `scripts/wsl/fetch_models.sh`) has no
    `refs/main` in the HF cache and resolving the implicit `main` revision
    fails under `HF_HUB_OFFLINE=1` even though the files are present.

    An explicit literal `tokenizer_name` (e.g. "gpt2") has no corresponding
    revision field in `VerbalizerYamlConfig`, so there is nothing to pin it
    to; it resolves against whatever `main` means in the local environment,
    same as before this fix.
    """
    if verb_config.tokenizer_name is not None:
        return None
    return encoder_revision


def _build_encode_cache_meta(
    llm_config: LLMConfig, verbalizer: str, *, deterministic: bool
) -> dict[str, Any]:
    import transformers

    meta: dict[str, Any] = {
        "verbalizer": verbalizer,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": llm_config.model_id,
        "revision": llm_config.revision,
        "dtype": llm_config.dtype,
        "quantize": llm_config.quantize,
        "pooling": llm_config.pooling,
        "attn_implementation": llm_config.attn_implementation,
        "deterministic": deterministic,
        "encoder_version": ENCODER_VERSION,
        "batch_size": llm_config.batch_size,
        "max_batch_tokens": llm_config.max_batch_tokens,
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_sha"] = "unknown"
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
    return meta


def _build_cache_key(
    llm_config: LLMConfig,
    verbalizer: str,
    verb_config: VerbalizerYamlConfig,
) -> str:
    deterministic = _encode_deterministic(llm_config)
    return compute_cache_key(
        CacheKeyConfig(
            model_id=llm_config.model_id,
            revision=llm_config.revision,
            verbalizer_name=verbalizer,
            verbalizer_config=verb_config.model_dump(),
            max_len=llm_config.max_len,
            dtype=llm_config.dtype,
            quantize=llm_config.quantize,
            pooling=llm_config.pooling,
            attn_implementation=llm_config.attn_implementation,
            deterministic=deterministic,
            encoder_version=ENCODER_VERSION,
        )
    )


def _sample_from_row(row: dict[str, Any]) -> Sample:
    return Sample(
        user_id=int(row["user_id"]),
        cutoff_ts=int(row["cutoff_ts"]),
        target_item=int(row["target_item"]),
        history=[int(h) for h in row["history"]],
        is_repeat=bool(row["is_repeat"]),
        target_is_cold=bool(row["target_is_cold"]),
    )


class TokenizerResolutionError(RuntimeError):
    """Raised when the tokenizer named by a verbalizer/model config can't be loaded."""


def _resolve_verbalizer_and_budget(
    verb_config: VerbalizerYamlConfig,
    llm_config: LLMConfig,
) -> tuple[TemplateVerbalizer, TokenBudget]:
    """Build the renderer + token budget shared by `verbalize render` and `encode run`.

    Both commands MUST resolve the tokenizer identically, or a human reviewing
    `verbalize render --n 20` output (DESIGN.md §9 M2's explicit acceptance step)
    would be looking at prompts different from what `encode run` actually feeds
    to the model. Keeping this logic in one place is what prevents the two
    commands from drifting apart again.
    """
    tokenizer_name = _resolve_tokenizer_name(verb_config, llm_config.model_id)
    revision = _resolve_tokenizer_revision(verb_config, llm_config.revision)
    try:
        get_tokenizer(tokenizer_name, revision)
    except OSError as exc:
        raise TokenizerResolutionError(
            f"Could not load tokenizer '{tokenizer_name}' (revision={revision!r}) for "
            f"verbalizer '{verb_config.name}' (resolved from tokenizer_name: "
            f"{verb_config.tokenizer_name!r}, model: {llm_config.model_id!r}). "
            "This usually means either the tokenizer needs to be downloaded from "
            "the Hugging Face Hub but the environment is offline "
            "(HF_HUB_OFFLINE=1 / GENREC_NO_NETWORK=1 is set, or there is no network "
            "access), or the tokenizer id is wrong/gated without credentials."
        ) from exc
    renderer = build_verbalizer_from_config(verb_config)
    budget = TokenBudget(
        max_tokens=verb_config.max_tokens, tokenizer_name=tokenizer_name, revision=revision
    )
    return renderer, budget


def _render_texts(
    samples: pl.DataFrame,
    items: pl.DataFrame,
    users: pl.DataFrame,
    interactions: pl.DataFrame,
    renderer: TemplateVerbalizer,
    budget: TokenBudget,
) -> list[str]:
    """Render one prompt per sample row, in row order."""
    return [
        renderer.render(_sample_from_row(row), items, users, interactions, budget)
        for row in samples.iter_rows(named=True)
    ]


@verbalize_app.command("render")
def verbalize_render(
    dataset: str = typer.Option(..., "--dataset", help="Dataset config name"),
    model: str = typer.Option(
        "qwen3-1.7b-base",
        "--model",
        help=(
            "LLM config name. Determines the tokenizer used to budget/truncate "
            "prompts when the verbalizer's tokenizer_name is null, exactly as "
            "`encode run` does, so the two commands never emit different prompts "
            "for the same sample."
        ),
    ),
    verbalizer: str = typer.Option("v1_full", "--verbalizer", help="Verbalizer config name"),
    n: int = typer.Option(20, "--n", help="Number of samples to dump"),
    output: str = typer.Option("reports/verbalizer_samples.md", "--output"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render verbalizer samples to markdown for human review."""
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
    try:
        renderer, budget = _resolve_verbalizer_and_budget(verb_config, llm_config)
    except TokenizerResolutionError as exc:
        console.print(f"[red]Tokenizer resolution failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    subset = samples.head(n)
    texts = _render_texts(subset, items, users, interactions, renderer, budget)

    lines = [f"# Verbalizer samples ({verbalizer})", ""]
    for i, prompt in enumerate(texts, start=1):
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
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            help="eval encodes samples.parquet; train encodes train_samples.parquet",
        ),
    ] = "eval",
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

    if scope not in ("eval", "train"):
        console.print(f"[red]Invalid scope:[/red] {scope!r} (expected 'eval' or 'train')")
        raise typer.Exit(code=1)
    cache_scope: CacheScope = scope  # type: ignore[assignment]

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    if scope == "train":
        encode_samples = read_train_samples(data_dir)
    else:
        encode_samples = samples
    try:
        renderer, budget = _resolve_verbalizer_and_budget(verb_config, llm_config)
    except TokenizerResolutionError as exc:
        console.print(f"[red]Tokenizer resolution failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    texts = _render_texts(encode_samples, items, users, interactions, renderer, budget)
    sample_ids = [int(x) for x in encode_samples["sample_id"].to_list()]

    encoder = PrefillEncoder.from_config(llm_config)
    hidden_dim = int(encoder.encode_batch([texts[0]]).shape[1])
    cache_key = _build_cache_key(llm_config, verbalizer, verb_config)
    cache = HiddenStateCache(root / cache_dir, cache_key, len(texts), hidden_dim, scope=cache_scope)
    if cache.exists():
        console.print(f"[yellow]Cache hit:[/yellow] {cache.memmap_path}")
        raise typer.Exit(code=0)

    completed = cache.completed_rows()
    if completed:
        console.print(
            f"[yellow]Resuming[/yellow] encode: {len(completed)}/{len(texts)} rows already cached"
        )

    lengths = encoder.token_lengths(texts)
    deterministic = _encode_deterministic(llm_config)
    if deterministic:
        batch_size = llm_config.batch_size or 8
        batch_groups = [
            list(range(start, min(start + batch_size, len(texts))))
            for start in range(0, len(texts), batch_size)
        ]
    else:
        batch_groups = batch_indices_by_token_budget(
            lengths,
            llm_config.max_batch_tokens,
            llm_config.batch_size,
        )

    for batch_indices in batch_groups:
        pending = [idx for idx in batch_indices if idx not in completed]
        if not pending:
            continue
        batch_texts = [texts[i] for i in pending]
        batch_hidden = encoder.encode_batch(batch_texts)
        cache.write_rows(pending, batch_hidden)

    cache.finalize(
        sample_ids,
        _build_encode_cache_meta(llm_config, verbalizer, deterministic=deterministic),
    )
    msg = (
        f"[green]Cached[/green] {len(texts)} vectors "
        f"({cache.expected_bytes} bytes) -> {cache.memmap_path}"
    )
    console.print(msg)


@train_app.command("head")
def train_head(
    exp: str = typer.Option(..., "--exp", help="M3 experiment config name"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train GenRecLite ranking head on cached hidden states."""
    _setup_logging(verbose)
    root = find_project_root()
    exp_config = load_m3_exp_config(exp, config_dir=root / "configs")
    data_config = load_data_config(exp_config.dataset, config_dir=root / "configs")
    llm_config = load_llm_config(exp_config.model, config_dir=root / "configs")
    verb_config = load_verbalizer_config(exp_config.verbalizer, config_dir=root / "configs")

    run_seed = seed if seed is not None else exp_config.seeds[0]
    seeds_to_run = [run_seed] if seed is not None else list(exp_config.seeds)
    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(code=1)

    interactions, items, _, samples = read_parquet_bundle(data_dir)
    try:
        train_samples = read_train_samples(data_dir)
    except FileNotFoundError as exc:
        console.print(
            f"[red]Train samples not found:[/red] {exc}\n"
            "Run `python -m genrec_lite data prepare --dataset "
            f"{exp_config.dataset}` first."
        )
        raise typer.Exit(code=1) from exc

    cache_key = _build_cache_key(llm_config, exp_config.verbalizer, verb_config)
    cache_dir = root / exp_config.cache_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        eval_store = HiddenStateStore.from_cache_dir(cache_dir, cache_key, scope="eval")
        train_store = HiddenStateStore.from_cache_dir(cache_dir, cache_key, scope="train")
    except FileNotFoundError as exc:
        console.print(
            f"[red]Hidden state cache missing:[/red] {exc}\n"
            "Run `encode run --scope eval` and `encode run --scope train` first."
        )
        raise typer.Exit(code=1) from exc

    hidden_dim = eval_store.hidden_dim
    item_init, freeze = build_item_init_matrix(items, exp_config.item_init, exp_config.embed_model)
    log_q = build_log_q(interactions, items.height)

    all_results: list[pd.DataFrame] = []
    for current_seed in seeds_to_run:
        _set_seed(current_seed)
        seed_model = GenRecLite(
            d_llm=hidden_dim,
            d_emb=exp_config.head.d_emb,
            n_items=items.height,
            scorer=exp_config.head.scorer,
            dropout=exp_config.head.dropout,
            item_init=item_init,
        )
        apply_item_init_freeze(seed_model, freeze)
        trainer = HeadTrainer(
            model=seed_model,
            train_store=train_store,
            eval_store=eval_store,
            train_samples=train_samples,
            valid_samples=samples,
            items=items,
            interactions=interactions,
            config=exp_config.train_head,
            device=device,
            log_q=log_q,
            ks=tuple(exp_config.ks),
            eval_batch_size=exp_config.eval_batch_size,
            cold_threshold=exp_config.cold_threshold,
            method_name="genrec_lite",
        )
        console.print(f"[bold]Training head[/bold] (seed={current_seed})")
        trainer.fit()
        eval_samples = samples.filter(pl.col("split") == exp_config.eval_split)
        result = evaluate(
            score_fn=trainer.score_batch,
            samples=eval_samples,
            items=items,
            interactions=interactions,
            ks=tuple(exp_config.ks),
            cold_threshold=exp_config.cold_threshold,
            method="genrec_lite",
            eval_batch_size=exp_config.eval_batch_size,
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


if __name__ == "__main__":
    app()
