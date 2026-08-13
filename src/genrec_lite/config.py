"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

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


class LLMConfig(BaseModel):
    model_id: str
    revision: str = "main"
    license: str
    commercial_use_ok: bool = True
    dtype: str = "bfloat16"
    pooling: Literal["last", "mean", "eos"] = "last"
    max_len: int = 512
    quantize: str | None = None


class VerbalizerYamlConfig(BaseModel):
    name: str
    variant: str = "v1_full"
    include_context: bool = True
    include_descriptions: bool = True
    max_history: int = 20
    desc_top_k: int = 3
    title_max_chars: int = 60
    max_tokens: int = 512
    tokenizer_name: str = "gpt2"


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


def load_llm_config(
    model: str,
    config_dir: Path | None = None,
) -> LLMConfig:
    root = config_dir or Path("configs")
    model_path = root / "model" / "llm" / f"{model}.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"LLM config not found: {model_path}")
    data = _load_yaml(model_path)
    if "license" not in data:
        raise ValueError(f"LLM config must define license: {model_path}")
    return LLMConfig.model_validate(data)


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
