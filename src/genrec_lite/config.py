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


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by looking for pyproject.toml."""
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current
