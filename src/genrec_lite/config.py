"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

SplitStrategy = Literal["leave_one_out", "global_temporal"]


class DataConfig(BaseModel):
    dataset: str
    category: str = "Video_Games"
    split_strategy: SplitStrategy = "global_temporal"
    cold_threshold: int = Field(default=5, ge=1)
    output_dir: Path = Path("data/processed/amazon_video_games")
    seed: int = 42


class BaseConfig(BaseModel):
    seed: int = 42
    cold_threshold: int = 5
    split_strategy: SplitStrategy = "global_temporal"


class SasrecConfig(BaseModel):
    hidden_dim: int = 32
    num_layers: int = 1
    num_heads: int = 2
    max_seq_len: int = 50
    epochs: int = 3
    lr: float = 0.001
    batch_size: int = 64


class ExpConfig(BaseModel):
    dataset: str
    baselines: list[str] = Field(default_factory=lambda: ["pop"])
    ks: list[int] = Field(default_factory=lambda: [10, 20])
    seeds: list[int] = Field(default_factory=lambda: [0])
    eval_split: int = Field(default=2, ge=0, le=2)
    cold_threshold: int = Field(default=5, ge=1)
    sasrec: SasrecConfig = Field(default_factory=SasrecConfig)


class HeadConfig(BaseModel):
    d_emb: int = 256
    scorer: Literal["dot", "mlp"] = "dot"
    dropout: float = 0.1


class TrainHeadConfig(BaseModel):
    lr: float = 1e-3
    item_emb_lr: float = 1e-3
    batch_size: int = 512
    n_negatives: int = 4096
    epochs: int = 50
    early_stop_patience: int = 5
    monitor: str = "valid/ndcg@20"


class M3ExpConfig(BaseModel):
    dataset: str
    model: str
    verbalizer: str
    head: HeadConfig = Field(default_factory=HeadConfig)
    train_head: TrainHeadConfig = Field(default_factory=TrainHeadConfig)
    item_init: Literal["random", "text", "text_frozen"] = "text"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "cache/hidden_states"
    ks: list[int] = Field(default_factory=lambda: [10, 20])
    seeds: list[int] = Field(default_factory=lambda: [0])
    eval_split: int = Field(default=1, ge=0, le=2)
    cold_threshold: int = Field(default=5, ge=1)


class LLMConfig(BaseModel):
    model_id: str
    revision: str
    license: str
    commercial_use_ok: bool = True
    dtype: str = "bfloat16"
    pooling: Literal["last", "mean", "eos"] = "last"
    max_len: int = 512
    quantize: str | None = None
    attn_implementation: Literal["auto", "sdpa", "flash_attention_2", "eager"] = "auto"
    batch_size: int | None = None
    max_batch_tokens: int = 4096
    bnb_compute_dtype: str = "bfloat16"
    device: str = "auto"
    low_cpu_mem_usage: bool = True
    deterministic: bool = False
    trust_remote_code: bool = False
    gated: bool = False
    allow_floating_revision: bool = False

    @model_validator(mode="after")
    def revision_must_be_pinned(self) -> Self:
        if self.allow_floating_revision:
            return self
        if self.revision.strip().lower() in {"main", "master"}:
            msg = (
                f"revision '{self.revision}' is a moving branch alias, not a pinned "
                "commit/tag (DESIGN.md §2.4.4). Pin a concrete revision or set "
                "allow_floating_revision: true explicitly."
            )
            raise ValueError(msg)
        return self


class VerbalizerYamlConfig(BaseModel):
    name: str
    variant: str = "v1_full"
    include_context: bool = True
    include_descriptions: bool = True
    max_history: int = 20
    desc_top_k: int = 3
    title_max_chars: int = 60
    max_tokens: int = 512
    tokenizer_name: str | None = None


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def load_data_config(
    dataset: str,
    config_dir: Path | None = None,
) -> DataConfig:
    root = config_dir or Path("configs")
    base_path = root / "base.yaml"
    data_path = root / "data" / f"{dataset}.yaml"

    merged: dict[str, object] = {}
    if base_path.exists():
        merged.update(_load_yaml(base_path))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}. "
            f"Available datasets should be defined in configs/data/"
        )
    merged.update(_load_yaml(data_path))
    merged["dataset"] = dataset
    if "output_dir" in merged:
        merged["output_dir"] = Path(str(merged["output_dir"]))
    return DataConfig.model_validate(merged)


def load_exp_config(
    exp: str,
    config_dir: Path | None = None,
) -> ExpConfig:
    root = config_dir or Path("configs")
    exp_path = root / "exp" / f"{exp}.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    data = _load_yaml(exp_path)
    return ExpConfig.model_validate(data)


def load_m3_exp_config(
    exp: str,
    config_dir: Path | None = None,
) -> M3ExpConfig:
    root = config_dir or Path("configs")
    exp_path = root / "exp" / f"{exp}.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"M3 experiment config not found: {exp_path}")
    data = _load_yaml(exp_path)
    return M3ExpConfig.model_validate(data)


def validate_llm_config_file(path: Path) -> LLMConfig:
    """Validate a single model YAML against LLMConfig and §2.4.4 contract."""
    data = _load_yaml(path)
    if "license" not in data:
        raise ValueError(f"LLM config must define license: {path}")
    if "commercial_use_ok" not in data:
        raise ValueError(f"LLM config must define commercial_use_ok: {path}")
    try:
        return LLMConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_llm_config(
    model: str,
    config_dir: Path | None = None,
) -> LLMConfig:
    root = config_dir or Path("configs")
    model_path = root / "model" / "llm" / f"{model}.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"LLM config not found: {model_path}")
    return validate_llm_config_file(model_path)


def load_verbalizer_config(
    verbalizer: str,
    config_dir: Path | None = None,
) -> VerbalizerYamlConfig:
    root = config_dir or Path("configs")
    verb_path = root / "verbalizer" / f"{verbalizer}.yaml"
    if not verb_path.exists():
        raise FileNotFoundError(f"Verbalizer config not found: {verb_path}")
    data = _load_yaml(verb_path)
    data["name"] = verbalizer
    return VerbalizerYamlConfig.model_validate(data)


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by looking for pyproject.toml."""
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current
