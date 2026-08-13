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


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by looking for pyproject.toml."""
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current
