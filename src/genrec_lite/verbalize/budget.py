"""Token budget counting with tokenizer (DESIGN.md §5.2)."""

from __future__ import annotations

from functools import lru_cache

from transformers import PreTrainedTokenizerBase


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(text: str, tokenizer_name: str) -> int:
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_to_budget(text: str, max_tokens: int, tokenizer_name: str) -> str:
    tokenizer = get_tokenizer(tokenizer_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokens[:max_tokens]
    return str(tokenizer.decode(truncated, skip_special_tokens=True))
