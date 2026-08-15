"""GenRec-lite head trainer (DESIGN.md §7)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

from genrec_lite.config import TrainHeadConfig
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.genrec_lite import GenRecLite
from genrec_lite.train.hidden_store import HiddenStateStore
from genrec_lite.train.losses import sampled_softmax_with_logq

logger = logging.getLogger(__name__)


def build_log_q(interactions: pl.DataFrame, n_items: int) -> Tensor:
    """Popularity log-probabilities for sampled-softmax logQ correction."""
    train = interactions.filter(pl.col("split") == 0)
    counts = np.ones(n_items, dtype=np.float64)
    if train.height > 0:
        grouped = train.group_by("item_id").len()
        for row in grouped.iter_rows():
            item_id, count = int(row[0]), float(row[1])
            if 0 <= item_id < n_items:
                counts[item_id] = count
    probs = counts / counts.sum()
    return torch.from_numpy(np.log(probs)).float()


def parse_monitor(monitor: str) -> tuple[int, str]:
    """Parse monitor string like ``valid/ndcg@20`` into split id and metric name."""
    if "/" not in monitor:
        msg = f"Invalid monitor format: {monitor!r} (expected 'split/metric')"
        raise ValueError(msg)
    split_name, metric = monitor.split("/", 1)
    split_map = {"train": 0, "valid": 1, "test": 2}
    if split_name not in split_map:
        msg = f"Unknown split in monitor: {split_name!r}"
        raise ValueError(msg)
    return split_map[split_name], metric


@dataclass
class HeadTrainer:
    """Train GenRecLite on cached hidden states with sampled softmax."""

    model: GenRecLite
    train_store: HiddenStateStore
    eval_store: HiddenStateStore
    train_samples: pl.DataFrame
    valid_samples: pl.DataFrame
    items: pl.DataFrame
    interactions: pl.DataFrame
    config: TrainHeadConfig
    device: torch.device
    log_q: Tensor
    ks: tuple[int, ...] = (10, 20)
    cold_threshold: int = 5
    method_name: str = "genrec_lite"

    def __post_init__(self) -> None:
        self.model = self.model.to(self.device)
        self.log_q = self.log_q.to(self.device)
        self._best_state: dict[str, Tensor] | None = None
        self._best_metric = float("-inf")

    def fit(self) -> GenRecLite:
        """Run training with validation-based early stopping."""
        n_items = self.items.height
        n_negatives = min(self.config.n_negatives, max(n_items - 1, 0))
        monitor_split, monitor_metric = parse_monitor(self.config.monitor)

        head_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and not name.endswith("item_emb.weight")
        ]
        item_params = [p for p in self.model.head.item_emb.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.config.lr},
                {"params": item_params, "lr": self.config.item_emb_lr},
            ]
        )

        patience_left = self.config.early_stop_patience
        sample_ids = self.train_samples["sample_id"].to_list()
        targets = self.train_samples["target_item"].to_list()

        for epoch in range(self.config.epochs):
            self.model.train()
            perm = torch.randperm(len(sample_ids)).tolist()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), self.config.batch_size):
                batch_idx = perm[start : start + self.config.batch_size]
                batch_sample_ids = [int(sample_ids[i]) for i in batch_idx]
                batch_targets = [int(targets[i]) for i in batch_idx]
                h = self.train_store.get_vectors(batch_sample_ids).to(self.device)
                candidate_ids, target_positions = self._sample_candidates(
                    batch_targets,
                    n_negatives,
                    n_items,
                )
                scores = self.model.score(h, candidate_ids=candidate_ids)
                batch_log_q = self.log_q[candidate_ids]
                loss = sampled_softmax_with_logq(scores, target_positions, log_q=batch_log_q)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            metric_value = self._validation_metric(monitor_split, monitor_metric)
            logger.info(
                "epoch=%d loss=%.4f %s=%.4f",
                epoch + 1,
                avg_loss,
                self.config.monitor,
                metric_value,
            )

            if metric_value > self._best_metric:
                self._best_metric = metric_value
                self._best_state = copy.deepcopy(self.model.state_dict())
                patience_left = self.config.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.model.eval()
        return self.model

    def _sample_candidates(
        self,
        batch_targets: list[int],
        n_negatives: int,
        n_items: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(batch_targets)
        if n_items <= 1:
            candidates = torch.tensor(batch_targets, dtype=torch.long).unsqueeze(1)
            return candidates.to(self.device), torch.zeros(batch_size, dtype=torch.long)

        probs = torch.exp(self.log_q).clone()
        max_k = min(n_negatives + len(batch_targets), n_items)
        candidate_rows: list[list[int]] = []
        target_positions: list[int] = []

        for batch_index, target in enumerate(batch_targets):
            row_candidates = [target]
            in_batch_negs = {
                item for item in batch_targets if item != target and item not in row_candidates
            }
            row_candidates.extend(sorted(in_batch_negs))

            remaining = max_k - len(row_candidates)
            if remaining > 0:
                sample_probs = probs.clone()
                for chosen in row_candidates:
                    sample_probs[chosen] = 0.0
                total = sample_probs.sum()
                if total > 0:
                    sample_probs = sample_probs / total
                    extra = torch.multinomial(sample_probs, remaining, replacement=False).tolist()
                    row_candidates.extend(int(x) for x in extra)

            target_positions.append(row_candidates.index(target))
            candidate_rows.append(row_candidates)

        max_width = max(len(row) for row in candidate_rows)
        padded = []
        for row in candidate_rows:
            padded.append(row + [row[-1]] * (max_width - len(row)))
        candidate_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        target_idx = torch.tensor(target_positions, dtype=torch.long, device=self.device)
        return candidate_ids, target_idx

    def _validation_metric(self, split_id: int, metric_name: str) -> float:
        samples = self.valid_samples.filter(pl.col("split") == split_id)
        if samples.height == 0:
            return float("-inf")
        result = evaluate(
            score_fn=self.score_batch,
            samples=samples,
            items=self.items,
            interactions=self.interactions,
            ks=self.ks,
            slices=("all",),
            cold_threshold=self.cold_threshold,
            method=self.method_name,
        )
        if result.empty or metric_name not in result.columns:
            return float("-inf")
        value = result.iloc[0][metric_name]
        return float(value) if value == value else float("-inf")

    def score_batch(self, samples: pl.DataFrame) -> np.ndarray:
        """Score all catalog items for cached eval samples."""
        self.model.eval()
        sample_ids = [int(x) for x in samples["sample_id"].to_list()]
        h = self.eval_store.get_vectors(sample_ids).to(self.device)
        with torch.no_grad():
            scores = self.model.score(h)
        return scores.cpu().numpy()
