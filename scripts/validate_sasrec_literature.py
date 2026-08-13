#!/usr/bin/env python3
"""Validate SASRec against literature values (DESIGN.md §8.4, manual/slow)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from genrec_lite.config import find_project_root, load_data_config, load_exp_config
from genrec_lite.data.schema import read_parquet_bundle
from genrec_lite.eval.runner import evaluate
from genrec_lite.models.baselines import build_baseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Placeholder literature reference values for Video_Games 5-core last_out.
# Update with published numbers after manual verification.
LITERATURE_NDCG20 = 0.05
TOLERANCE = 0.20


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SASRec literature alignment")
    parser.add_argument("--exp", default="m1_baselines")
    parser.add_argument("--dataset", default="amazon_video_games")
    args = parser.parse_args()

    root = find_project_root()
    exp_config = load_exp_config(args.exp, config_dir=root / "configs")
    data_config = load_data_config(exp_config.dataset, config_dir=root / "configs")
    data_dir = root / data_config.output_dir
    if not data_dir.exists():
        logger.error("Processed data not found at %s. Run data prepare first.", data_dir)
        return 1

    interactions, items, users, samples = read_parquet_bundle(data_dir)
    test_samples = samples.filter(pl.col("split") == exp_config.eval_split)

    model = build_baseline("sasrec", sasrec_config=exp_config.sasrec)
    model.fit(interactions, items)
    result = evaluate(
        score_fn=model.score_batch,
        samples=test_samples,
        items=items,
        interactions=interactions,
        ks=(20,),
        slices=("all",),
        cold_threshold=exp_config.cold_threshold,
        method="sasrec",
    )
    ndcg = float(result.iloc[0]["ndcg@20"])
    ratio = ndcg / LITERATURE_NDCG20 if LITERATURE_NDCG20 > 0 else 0.0
    logger.info("SASRec NDCG@20=%.4f (literature=%.4f, ratio=%.2f)", ndcg, LITERATURE_NDCG20, ratio)
    if abs(ratio - 1.0) > TOLERANCE:
        logger.warning("Outside ±20%% tolerance. Check split/eval protocol.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
