"""Text embedding tests."""

from __future__ import annotations

from genrec_lite.encode.text_embed import _hash_embed, build_text_embed_fn


def test_hash_embed_is_deterministic() -> None:
    texts = ["hello", "world"]
    a = _hash_embed(texts)
    b = _hash_embed(texts)
    assert (a == b).all()


def test_build_text_embed_fn_without_sentence_transformers() -> None:
    embed_fn = build_text_embed_fn()
    vectors = embed_fn(["item one", "item two"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
