"""Verbalizer tests (DESIGN.md §14.2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from genrec_lite.config import DataConfig, LLMConfig, VerbalizerYamlConfig, load_verbalizer_config
from genrec_lite.verbalize.base import Sample, TokenBudget
from genrec_lite.verbalize.budget import count_tokens, get_tokenizer
from genrec_lite.verbalize.compress import CompressConfig, compress_history_events
from genrec_lite.verbalize.templates import build_verbalizer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
VERBALIZER_CONFIG_DIR = CONFIG_DIR / "verbalizer"
VERBALIZER_CONFIG_PATHS = sorted(VERBALIZER_CONFIG_DIR.glob("*.yaml"))


def _mini_sample() -> Sample:
    return Sample(
        user_id=0,
        cutoff_ts=1_600_100_000,
        target_item=1,
        history=[0, 1, 2],
        is_repeat=False,
        target_is_cold=False,
    )


def _mini_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    items = pl.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["Game A", "Game B", "Game C"],
            "brand": ["BrandA", "BrandB", "BrandC"],
            "category_path": ["Electronics > Games > A"] * 3,
            "price": [10.0, 20.0, 30.0],
            "description": ["Desc A", "Desc B", "Desc C"],
            "first_seen_ts": [1_600_000_000, 1_600_010_000, 1_600_020_000],
            "n_train_inter": [10, 10, 10],
        }
    )
    users = pl.DataFrame(
        {
            "user_id": [0],
            "raw_id": ["user_0"],
            "n_inter": [3],
            "first_ts": [1_600_000_000],
            "last_ts": [1_600_100_000],
            "repeat_ratio": [0.1],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": [0, 0, 0],
            "item_id": [0, 1, 2],
            "ts": [1_600_000_000, 1_600_050_000, 1_600_090_000],
            "split": [0, 0, 0],
        }
    )
    return items, users, interactions


def test_render_deterministic(tiny_model_id: str) -> None:
    sample = _mini_sample()
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v1_full")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    out1 = verbalizer.render(sample, items, users, interactions, budget)
    out2 = verbalizer.render(sample, items, users, interactions, budget)
    assert out1 == out2


def test_compression_reduces_below_budget(tiny_model_id: str) -> None:
    events = [
        {
            "item_id": i,
            "title": f"Very Long Game Title Number {i}" * 3,
            "category_leaf": "Games",
            "price": 10.0,
            "description": "Long description " * 20,
            "rating": 3.0,
        }
        for i in range(30)
    ]
    compressed = compress_history_events(events, CompressConfig(max_history=5, desc_top_k=0))
    assert len(compressed) <= 5
    text = "\n".join(str(e["title"]) for e in compressed)
    assert count_tokens(text, tiny_model_id) <= 256


def test_compression_priority_order() -> None:
    events = [
        {"item_id": 1, "title": "A", "rating": 1.0, "price": 1.0, "description": "d"},
        {"item_id": 2, "title": "B", "rating": 5.0, "price": 50.0, "description": "d"},
    ]
    cfg = CompressConfig(max_history=10, min_rating=4.0, min_price=10.0)
    compressed = compress_history_events(events, cfg)
    assert len(compressed) == 1
    assert compressed[0]["item_id"] == 2


def test_id_only_verbalizer_contains_no_titles(tiny_model_id: str) -> None:
    sample = _mini_sample()
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v0_ids_only")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    prompt = verbalizer.render(sample, items, users, interactions, budget)
    assert "Game A" not in prompt
    assert "item_0" in prompt


def test_prompt_prefix_is_shared(tiny_model_id: str) -> None:
    items, users, interactions = _mini_tables()
    verbalizer = build_verbalizer("v1_full")
    budget = TokenBudget(max_tokens=512, tokenizer_name=tiny_model_id)
    history = [0, 1, 2]
    cutoff = 1_600_100_000
    sample_a = Sample(0, cutoff, 1, history, False, False)
    sample_b = Sample(0, cutoff, 2, history, True, False)
    prompt_a = verbalizer.render(sample_a, items, users, interactions, budget)
    prompt_b = verbalizer.render(sample_b, items, users, interactions, budget)
    tokenizer = get_tokenizer(tiny_model_id)
    prefix_a = tokenizer.encode(prompt_a, add_special_tokens=False)[:100]
    prefix_b = tokenizer.encode(prompt_b, add_special_tokens=False)[:100]
    assert prefix_a == prefix_b


# --- tokenizer_name resolution (issue: budget must be counted with the same
# tokenizer that actually encodes the prompt, DESIGN.md §5.2) -----------------


def test_resolve_tokenizer_name_uses_encoder_when_null() -> None:
    """`tokenizer_name: null` in the verbalizer YAML means 'use the encoder's own
    tokenizer' -- this is what makes the §5.2 token budget match what `encode
    run` actually feeds to the model."""
    from genrec_lite.cli import _resolve_tokenizer_name

    cfg = VerbalizerYamlConfig(name="v1_full", tokenizer_name=None)
    assert _resolve_tokenizer_name(cfg, "Qwen/Qwen3-1.7B-Base") == "Qwen/Qwen3-1.7B-Base"


def test_resolve_tokenizer_name_prefers_explicit_literal() -> None:
    """An explicit tokenizer_name always wins over the encoder's model id."""
    from genrec_lite.cli import _resolve_tokenizer_name

    cfg = VerbalizerYamlConfig(name="v1_full", tokenizer_name="gpt2")
    assert _resolve_tokenizer_name(cfg, "Qwen/Qwen3-1.7B-Base") == "gpt2"


def test_resolve_tokenizer_name_falls_back_to_gpt2_without_encoder() -> None:
    """No verbalizer tokenizer_name and no encoder model id: last-resort default."""
    from genrec_lite.cli import _resolve_tokenizer_name

    cfg = VerbalizerYamlConfig(name="v1_full", tokenizer_name=None)
    assert _resolve_tokenizer_name(cfg) == "gpt2"


def test_resolve_tokenizer_revision_uses_encoder_revision_when_null() -> None:
    """`tokenizer_name: null` means "use the encoder's tokenizer", so the
    budget tokenizer must also be pinned to the *encoder's* revision
    (DESIGN.md §2.4.4) -- otherwise a revision-pinned download has no
    `refs/main` in the HF cache and resolving the implicit `main` revision
    fails under HF_HUB_OFFLINE=1."""
    from genrec_lite.cli import _resolve_tokenizer_revision

    cfg = VerbalizerYamlConfig(name="v1_full", tokenizer_name=None)
    assert _resolve_tokenizer_revision(cfg, "ea980cb0a6c2ae4b936e82123acc929f1cec04c1") == (
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    )


def test_resolve_tokenizer_revision_is_none_for_explicit_literal() -> None:
    """An explicit literal tokenizer_name (e.g. "gpt2") has no revision field in
    VerbalizerYamlConfig to pin it to, so there is nothing to thread through --
    it must resolve to None, not silently reuse the encoder's revision."""
    from genrec_lite.cli import _resolve_tokenizer_revision

    cfg = VerbalizerYamlConfig(name="v1_full", tokenizer_name="gpt2")
    assert _resolve_tokenizer_revision(cfg, "ea980cb0a6c2ae4b936e82123acc929f1cec04c1") is None


def test_get_tokenizer_passes_revision_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_tokenizer` must thread `revision` into `AutoTokenizer.from_pretrained`
    -- a revision-pinned download (scripts/wsl/fetch_models.sh) never writes
    `refs/main`, so omitting `revision=` fails offline even though the
    tokenizer files are present under the pinned snapshot directory."""
    get_tokenizer.cache_clear()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(_model_id: str, **kwargs: object) -> MagicMock:
        calls.append(dict(kwargs))
        return mock_tokenizer

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
        get_tokenizer("fake/pinned-model", "deadbeefcafe0123456789abcdef01234567890a")

    assert calls == [{"revision": "deadbeefcafe0123456789abcdef01234567890a"}]
    get_tokenizer.cache_clear()


def test_get_tokenizer_omits_revision_kwarg_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No revision pinned -> no `revision=` kwarg at all (matches
    `PrefillEncoder`'s `from_pretrained_common` pattern in
    `genrec_lite.encode.prefill`), so behavior for unpinned callers is
    unchanged from before this fix."""
    get_tokenizer.cache_clear()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(_model_id: str, **kwargs: object) -> MagicMock:
        calls.append(dict(kwargs))
        return mock_tokenizer

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
        get_tokenizer("fake/unpinned-model")

    assert calls == [{}]
    get_tokenizer.cache_clear()


def test_get_tokenizer_lru_cache_distinguishes_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two different revisions of the same tokenizer id must not collide in the
    `@lru_cache` -- otherwise loading a second revision for an id already in
    the cache would silently return the first revision's tokenizer."""
    get_tokenizer.cache_clear()
    load_count = 0

    def fake_from_pretrained(_model_id: str, **kwargs: object) -> MagicMock:
        nonlocal load_count
        load_count += 1
        tok = MagicMock()
        tok.pad_token = "<pad>"
        tok.revision = kwargs.get("revision")
        return tok

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
        tok_a = get_tokenizer("fake/shared-model-id", "revision-a")
        tok_b = get_tokenizer("fake/shared-model-id", "revision-b")
        tok_a_again = get_tokenizer("fake/shared-model-id", "revision-a")

    assert load_count == 2, "expected exactly one load per distinct (name, revision) pair"
    assert tok_a is not tok_b
    assert tok_a is tok_a_again
    assert tok_a.revision == "revision-a"
    assert tok_b.revision == "revision-b"
    get_tokenizer.cache_clear()


@pytest.mark.parametrize("path", VERBALIZER_CONFIG_PATHS, ids=lambda p: p.name)
def test_shipped_verbalizer_config_parses_and_is_null_tokenizer(path: Path) -> None:
    """Every shipped configs/verbalizer/*.yaml must parse, and (per the fix for
    the gpt2/Qwen3 budget-mismatch bug) must defer to the encoder's tokenizer
    rather than hardcoding gpt2."""
    cfg = load_verbalizer_config(path.stem, config_dir=CONFIG_DIR)
    assert cfg.tokenizer_name is None, (
        f"{path.name} sets tokenizer_name={cfg.tokenizer_name!r}; expected null "
        "so the §5.2 token budget is counted with the encoder's own tokenizer."
    )


@pytest.mark.parametrize("path", VERBALIZER_CONFIG_PATHS, ids=lambda p: p.name)
def test_shipped_verbalizer_config_builds_usable_budget(path: Path, tiny_model_id: str) -> None:
    """A TokenBudget built from each shipped config (resolved against a stand-in
    encoder model id) must actually be usable for counting tokens."""
    from genrec_lite.cli import _resolve_tokenizer_name

    cfg = load_verbalizer_config(path.stem, config_dir=CONFIG_DIR)
    tokenizer_name = _resolve_tokenizer_name(cfg, tiny_model_id)
    budget = TokenBudget(max_tokens=cfg.max_tokens, tokenizer_name=tokenizer_name)
    assert count_tokens("hello world", budget.tokenizer_name) > 0


def test_verbalize_render_and_encode_run_produce_identical_prompt(
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: Path,
    tmp_path: Path,
) -> None:
    """Regression guard for the `verbalize render` / `encode run` tokenizer-
    resolution drift (DESIGN.md §9 M2): both commands must resolve the same
    tokenizer for the same verbalizer + model, or a human reviewing
    `verbalize render --n 20` output would be looking at prompts different
    from what `encode run` actually feeds to the model.

    Network-free: `count_tokens`/`truncate_to_budget`/`get_tokenizer` are
    monkeypatched so this exercises only the tokenizer-name *routing* through
    the CLI (the actual bug), not real tokenizer downloads. This must fail
    before `verbalize render` gains a `--model` option: pre-fix it doesn't
    accept `model=`, and post-fix-but-pre-plumbing it would resolve "gpt2"
    while `encode run` resolves the real encoder id.
    """
    import genrec_lite.cli as cli_module
    import genrec_lite.verbalize.templates as templates_module

    resolved_names: list[str] = []
    resolved_revisions: list[str | None] = []

    def fake_count_tokens(text: str, tokenizer_name: str, revision: str | None = None) -> int:
        return 0  # never exceed budget -> render() never takes the retry path

    def fake_truncate_to_budget(
        text: str, max_tokens: int, tokenizer_name: str, revision: str | None = None
    ) -> str:
        resolved_names.append(tokenizer_name)
        resolved_revisions.append(revision)
        return f"{text}\n<<resolved-tokenizer:{tokenizer_name}>>"

    monkeypatch.setattr(templates_module, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(templates_module, "truncate_to_budget", fake_truncate_to_budget)
    monkeypatch.setattr(
        cli_module, "get_tokenizer", lambda name, revision=None: object(), raising=False
    )

    encoder_model_id = "fake/encoder-model-id"
    captured_texts: list[str] = []

    class _FakeEncoder:
        def token_lengths(self, texts: list[str]) -> list[int]:
            return [1] * len(texts)

        def encode_batch(self, texts: list[str]) -> torch.Tensor:
            captured_texts.extend(texts)
            return torch.zeros((len(texts), 4), dtype=torch.float32)

    class _FakePrefillEncoder:
        @classmethod
        def from_config(cls, llm_config: LLMConfig) -> _FakeEncoder:
            return _FakeEncoder()

    monkeypatch.setattr(cli_module, "PrefillEncoder", _FakePrefillEncoder)

    def fake_load_data_config(dataset: str, config_dir: Path | None = None) -> DataConfig:
        return DataConfig(dataset=dataset, output_dir=mini_dataset)

    def fake_load_llm_config(model: str, config_dir: Path | None = None) -> LLMConfig:
        return LLMConfig(
            model_id=encoder_model_id,
            revision="deadbeefcafe",
            license="MIT",
            deterministic=True,
        )

    def fake_load_verbalizer_config(
        verbalizer: str, config_dir: Path | None = None
    ) -> VerbalizerYamlConfig:
        # tokenizer_name=None simulates the fixed YAMLs regardless of what's on
        # disk, isolating this test to the CLI routing logic under test.
        return VerbalizerYamlConfig(name=verbalizer, max_tokens=100_000, tokenizer_name=None)

    monkeypatch.setattr(cli_module, "load_data_config", fake_load_data_config)
    monkeypatch.setattr(cli_module, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(cli_module, "load_verbalizer_config", fake_load_verbalizer_config)

    render_output = tmp_path / "samples.md"
    cli_module.verbalize_render(
        dataset="mini",
        model="mini-model",
        verbalizer="v1_full",
        n=1000,
        output=str(render_output),
        verbose=False,
    )
    cli_module.encode_run(
        dataset="mini",
        model="mini-model",
        verbalizer="v1_full",
        cache_dir=str(tmp_path / "cache"),
        verbose=False,
    )

    assert captured_texts, "encode run did not encode any samples"
    markdown = render_output.read_text(encoding="utf-8")
    marker = "## Sample 1"
    assert marker in markdown
    after_header = markdown[markdown.index(marker) :]
    rendered_prompt = after_header.split("```")[1].strip("\n")

    assert resolved_names, "truncate_to_budget was never called"
    assert all(name == encoder_model_id for name in resolved_names), (
        "expected every resolved tokenizer to be the encoder's model id "
        f"({encoder_model_id!r}), got {sorted(set(resolved_names))!r}"
    )
    assert resolved_revisions, "truncate_to_budget was never called"
    assert all(revision == "deadbeefcafe" for revision in resolved_revisions), (
        "expected every resolved tokenizer revision to be the encoder's pinned "
        f"revision ('deadbeefcafe'), got {sorted(set(resolved_revisions))!r} -- a "
        "budget tokenizer resolving to an unpinned 'main' would defeat the "
        "DESIGN.md §2.4.4 revision pin"
    )
    assert rendered_prompt == captured_texts[0]
