"""HeadTrainer tests (M3, DESIGN.md §7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from genrec_lite.config import TrainHeadConfig
from genrec_lite.data.schema import read_parquet_bundle, read_train_samples
from genrec_lite.data.split import SPLIT_VALID
from genrec_lite.encode.cache import HiddenStateCache
from genrec_lite.models.genrec_lite import GenRecLite
from genrec_lite.train.head_trainer import HeadTrainer, build_log_q
from genrec_lite.train.hidden_store import HiddenStateStore
from genrec_lite.train.item_init import apply_item_init_freeze, build_item_init_matrix


def _write_fake_cache(
    cache_dir: Path,
    key: str,
    sample_ids: list[int],
    hidden_dim: int,
    scope: str = "eval",
) -> HiddenStateCache:
    cache = HiddenStateCache(cache_dir, key, len(sample_ids), hidden_dim, scope=scope)  # type: ignore[arg-type]
    rng = np.random.default_rng(42)
    hidden = torch.from_numpy(rng.standard_normal((len(sample_ids), hidden_dim)).astype(np.float32))
    cache.save(sample_ids, hidden, meta={"test": True})
    return cache


@pytest.fixture
def head_trainer_bundle(mini_dataset: Path) -> dict[str, object]:
    interactions, items, _, samples = read_parquet_bundle(mini_dataset)
    train_samples = read_train_samples(mini_dataset)
    hidden_dim = 16
    cache_dir = mini_dataset / "cache"
    key = "test-key"
    eval_ids = [int(x) for x in samples["sample_id"].to_list()]
    train_ids = [int(x) for x in train_samples["sample_id"].to_list()]
    _write_fake_cache(cache_dir, key, eval_ids, hidden_dim, scope="eval")
    _write_fake_cache(cache_dir, key, train_ids, hidden_dim, scope="train")
    eval_store = HiddenStateStore.from_cache_dir(cache_dir, key, hidden_dim, scope="eval")
    train_store = HiddenStateStore.from_cache_dir(cache_dir, key, hidden_dim, scope="train")
    item_init, freeze = build_item_init_matrix(
        items, "text", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = GenRecLite(
        d_llm=hidden_dim,
        d_emb=8,
        n_items=items.height,
        item_init=item_init,
    )
    apply_item_init_freeze(model, freeze)
    valid_samples = samples.filter(pl.col("split") == SPLIT_VALID)
    config = TrainHeadConfig(
        batch_size=4,
        n_negatives=4,
        epochs=1,
        early_stop_patience=1,
        monitor="valid/ndcg@20",
    )
    trainer = HeadTrainer(
        model=model,
        train_store=train_store,
        eval_store=eval_store,
        train_samples=train_samples,
        valid_samples=valid_samples,
        items=items,
        interactions=interactions,
        config=config,
        device=torch.device("cpu"),
        log_q=build_log_q(interactions, items.height),
    )
    return {
        "trainer": trainer,
        "items": items,
        "samples": samples,
        "hidden_dim": hidden_dim,
    }


def test_head_trainer_fit_one_epoch(head_trainer_bundle: dict[str, object]) -> None:
    trainer = head_trainer_bundle["trainer"]
    assert isinstance(trainer, HeadTrainer)
    trained = trainer.fit()
    assert trained is trainer.model


def test_head_trainer_score_batch_shape(head_trainer_bundle: dict[str, object]) -> None:
    trainer = head_trainer_bundle["trainer"]
    samples = head_trainer_bundle["samples"]
    items = head_trainer_bundle["items"]
    assert isinstance(trainer, HeadTrainer)
    assert isinstance(samples, pl.DataFrame)
    assert isinstance(items, pl.DataFrame)
    trainer.fit()
    scores = trainer.score_batch(samples.head(3))
    assert scores.shape == (3, items.height)


def test_text_frozen_item_emb_no_grad() -> None:
    n_items, d_init, d_emb, d_llm = 6, 10, 4, 8
    item_init = torch.randn(n_items, d_init)
    model = GenRecLite(d_llm=d_llm, d_emb=d_emb, n_items=n_items, item_init=item_init)
    apply_item_init_freeze(model, True)
    h = torch.randn(2, d_llm, requires_grad=False)
    loss = model.score(h).sum()
    loss.backward()
    assert model.head.item_emb.weight.grad is None
    assert model.head.proj.weight.grad is not None
