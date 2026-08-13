"""Text embedding for item initialization and TextKNN (DESIGN.md §2.4.3, §6)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_EMBED_DIM = 16


def _hash_embed(texts: list[str], dim: int = HASH_EMBED_DIM) -> np.ndarray:
    """Deterministic hash-based embeddings for CI / no-deps fallback."""
    vectors = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for j in range(dim):
            vectors[i, j] = (digest[j % len(digest)] - 128) / 128.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _sentence_transformer_embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float64)


def build_text_embed_fn(
    model_name: str = DEFAULT_EMBED_MODEL,
    fallback_dim: int = HASH_EMBED_DIM,
) -> Callable[[list[str]], np.ndarray]:
    """Return an embed function; falls back to hash embed if sentence-transformers unavailable."""
    try:
        import sentence_transformers  # noqa: F401

        def embed_fn(texts: list[str]) -> np.ndarray:
            return _sentence_transformer_embed(texts, model_name)

        logger.info("Using sentence-transformers model: %s", model_name)
        return embed_fn
    except ImportError:
        logger.warning("sentence-transformers not installed; using deterministic hash embeddings")

        def hash_fn(texts: list[str]) -> np.ndarray:
            return _hash_embed(texts, dim=fallback_dim)

        return hash_fn
