"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from genrec_lite.config import load_data_config, load_m3_exp_config


def test_load_amazon_video_games_config() -> None:
    root = Path(__file__).parent.parent
    config = load_data_config("amazon_video_games", config_dir=root / "configs")
    assert config.dataset == "amazon_video_games"
    assert config.category == "Video_Games"
    assert config.split_strategy == "global_temporal"
    assert config.cold_threshold == 5


def test_load_missing_dataset_raises() -> None:
    root = Path(__file__).parent.parent
    with pytest.raises(FileNotFoundError):
        load_data_config("nonexistent_dataset", config_dir=root / "configs")


def test_load_m3_frozen_uses_qwen3_8b_nf4() -> None:
    root = Path(__file__).parent.parent
    exp = load_m3_exp_config("m3_frozen", config_dir=root / "configs")
    assert exp.dataset == "amazon_video_games"
    assert exp.model == "qwen3-8b-base"
    assert exp.verbalizer == "v1_full"
    assert exp.eval_batch_size == 32


def test_m3_eval_batch_size_must_be_positive(tmp_path: Path) -> None:
    root = Path(__file__).parent.parent
    exp_path = root / "configs" / "exp" / "m3_frozen.yaml"
    data = exp_path.read_text(encoding="utf-8").replace("eval_batch_size: 32", "eval_batch_size: 0")
    tmp_config = tmp_path
    (tmp_config / "exp").mkdir(parents=True, exist_ok=True)
    (tmp_config / "exp" / "invalid.yaml").write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="eval_batch_size"):
        load_m3_exp_config("invalid", config_dir=tmp_config)
