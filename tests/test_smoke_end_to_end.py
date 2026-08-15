"""End-to-end smoke test (M3, DESIGN.md §14.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from genrec_lite.config import (
    DataConfig,
    HeadConfig,
    LLMConfig,
    M3ExpConfig,
    TrainHeadConfig,
    VerbalizerYamlConfig,
)
from genrec_lite.verbalize.base import TokenBudget


@pytest.mark.slow
def test_smoke_encode_train_eval_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: Path,
    tmp_path: Path,
) -> None:
    """Mini fixture pipeline: encode (eval+train) -> train head -> metrics output."""
    import genrec_lite.cli as cli_module

    hidden_dim = 8
    encoder_model_id = "fake/encoder-model-id"

    class _FakeEncoder:
        def token_lengths(self, texts: list[str]) -> list[int]:
            return [1] * len(texts)

        def encode_batch(self, texts: list[str]) -> torch.Tensor:
            return torch.zeros((len(texts), hidden_dim), dtype=torch.float32)

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
        return VerbalizerYamlConfig(name=verbalizer, max_tokens=100_000, tokenizer_name="gpt2")

    monkeypatch.setattr(cli_module, "load_data_config", fake_load_data_config)
    monkeypatch.setattr(cli_module, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(cli_module, "load_verbalizer_config", fake_load_verbalizer_config)

    def fake_load_m3_exp_config(exp: str, config_dir: Path | None = None) -> M3ExpConfig:
        return M3ExpConfig(
            dataset="mini",
            model="fake-model",
            verbalizer="v1_full",
            head=HeadConfig(d_emb=8, scorer="dot", dropout=0.1),
            train_head=TrainHeadConfig(
                batch_size=4,
                n_negatives=4,
                epochs=1,
                early_stop_patience=1,
                monitor="valid/ndcg@20",
            ),
            cache_dir="cache/hidden_states",
            eval_split=2,
            seeds=[0],
        )

    monkeypatch.setattr(cli_module, "load_m3_exp_config", fake_load_m3_exp_config)
    monkeypatch.setattr(
        cli_module,
        "_resolve_verbalizer_and_budget",
        lambda verb_config, llm_config: (
            object(),
            TokenBudget(max_tokens=512, tokenizer_name="gpt2"),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_render_texts",
        lambda encode_samples, items, users, interactions, renderer, budget: (
            ["prompt"] * encode_samples.height
        ),
    )
    monkeypatch.setattr(cli_module, "find_project_root", lambda start=None: tmp_path)

    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    cache_dir = str(tmp_path / "cache" / "hidden_states")

    cli_module.encode_run(
        dataset="mini",
        model="fake-model",
        verbalizer="v1_full",
        cache_dir=cache_dir,
        scope="eval",
        verbose=False,
    )
    cli_module.encode_run(
        dataset="mini",
        model="fake-model",
        verbalizer="v1_full",
        cache_dir=cache_dir,
        scope="train",
        verbose=False,
    )
    cli_module.train_head(exp="m3_frozen", seed=0, verbose=False)

    runs_dir = reports_root / "runs"
    assert runs_dir.exists()
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert run_dirs, "expected a run directory under reports/runs"
    latest = sorted(run_dirs)[-1]
    assert (latest / "metrics.json").exists()
    assert (tmp_path / "reports" / "results.md").exists()
