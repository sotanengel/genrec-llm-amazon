"""Token budget counting with tokenizer (DESIGN.md §5.2)."""

from __future__ import annotations

from functools import lru_cache

from transformers import PreTrainedTokenizerBase


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_name: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    """Load (and cache) a tokenizer, optionally pinned to a specific `revision`.

    `revision` is part of the `lru_cache` key (it's a normal call argument), so
    two different revisions of the same `tokenizer_name` never collide in the
    cache -- each gets its own cached tokenizer instance.

    A pinned `revision` is threaded through so a revision-pinned download (see
    `scripts/wsl/fetch_models.sh`, which downloads via `hf download <id>
    --revision <sha>`) resolves correctly under `HF_HUB_OFFLINE=1`. A
    revision-pinned snapshot has no `refs/main` written into the HF cache, so
    resolving the implicit `main` revision (i.e. calling `from_pretrained`
    without `revision=`) fails offline even though the files are present on
    disk under the pinned snapshot directory.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, object] = {}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(text: str, tokenizer_name: str, revision: str | None = None) -> int:
    tokenizer = get_tokenizer(tokenizer_name, revision)
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_to_budget(
    text: str, max_tokens: int, tokenizer_name: str, revision: str | None = None
) -> str:
    tokenizer = get_tokenizer(tokenizer_name, revision)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokens[:max_tokens]
    return str(tokenizer.decode(truncated, skip_special_tokens=True))
