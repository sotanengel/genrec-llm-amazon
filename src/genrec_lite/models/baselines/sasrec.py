"""SASRec baseline (DESIGN.md §6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from genrec_lite.config import SasrecConfig


class _SASRecDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: pl.DataFrame, max_seq_len: int, n_items: int) -> None:
        self._samples = samples
        self._max_seq_len = max_seq_len
        self._n_items = n_items

    def __len__(self) -> int:
        return self._samples.height

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._samples.row(idx, named=True)
        history = [int(i) for i in row["history"]][-self._max_seq_len :]
        target = int(row["target_item"])
        seq = torch.zeros(self._max_seq_len, dtype=torch.long)
        if history:
            seq[-len(history) :] = torch.tensor(history, dtype=torch.long)
        return seq, target


class SASRecModel(nn.Module):
    """Self-attentive sequential recommendation model."""

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Return sequence representations [B, L, D]."""
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        padding_mask = seq == 0
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return cast(torch.Tensor, encoded)

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """Return scores for all items [B, n_items]."""
        reps = self.forward(seq)
        last_idx = seq.ne(0).sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(seq.size(0), device=seq.device)
        last_rep = reps[batch_idx, last_idx]
        item_embs = self.item_emb.weight[1 : self.n_items + 1]
        return last_rep @ item_embs.T


@dataclass
class SASRecRecommender:
    """Trainable SASRec baseline."""

    config: SasrecConfig
    _model: SASRecModel | None = None
    _n_items: int = 0
    _device: torch.device = torch.device("cpu")

    def fit(self, interactions: pl.DataFrame, items: pl.DataFrame) -> None:
        self._n_items = items.height
        train_samples = self._build_train_samples(interactions)
        if train_samples.height == 0:
            self._model = SASRecModel(
                n_items=self._n_items,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_len,
            ).to(self._device)
            return

        self._model = SASRecModel(
            n_items=self._n_items,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
        ).to(self._device)
        dataset = _SASRecDataset(train_samples, self.config.max_seq_len, self._n_items)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.lr)

        self._model.train()
        for _ in range(self.config.epochs):
            for seq, target in loader:
                seq = seq.to(self._device)
                target = target.to(self._device)
                scores = self._model.score_all_items(seq)
                loss = nn.functional.cross_entropy(scores, target)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

    def _build_train_samples(self, interactions: pl.DataFrame) -> pl.DataFrame:
        train = interactions.filter(pl.col("split") == 0).sort(["user_id", "ts"])
        rows: list[dict[str, object]] = []
        sample_id = 0
        for user_id in train["user_id"].unique().to_list():
            user_items = train.filter(pl.col("user_id") == user_id)["item_id"].to_list()
            history: list[int] = []
            for item_id in user_items:
                if history:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "user_id": int(user_id),
                            "cutoff_ts": 0,
                            "target_item": int(item_id),
                            "history": list(history),
                            "split": 0,
                            "is_repeat": int(item_id) in history,
                            "target_is_cold": False,
                        }
                    )
                    sample_id += 1
                history.append(int(item_id))
        if not rows:
            return pl.DataFrame(
                schema={
                    "sample_id": pl.Int64,
                    "user_id": pl.Int32,
                    "cutoff_ts": pl.Int64,
                    "target_item": pl.Int32,
                    "history": pl.List(pl.Int32),
                    "split": pl.Int8,
                    "is_repeat": pl.Boolean,
                    "target_is_cold": pl.Boolean,
                }
            )
        return pl.DataFrame(rows)

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.eval()
        dataset = _SASRecDataset(samples, self.config.max_seq_len, self._n_items)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for seq, _ in loader:
                seq = seq.to(self._device)
                scores = self._model.score_all_items(seq).cpu().numpy()
                all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)

    def name(self) -> str:
        return "sasrec"
