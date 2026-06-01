"""Pinned Llama-3.1 tokenizer: length matching (F1/F5), token balance, delta-L scoring.

The same tokenizer is used local-side (filters/balance) and on AWS (Stage 8), so
the same physical item segments identically everywhere (plan Section 6.0, 10.2).
"""
from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

from .config import TOKENIZER_REPO, get_hf_token


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_REPO, token=get_hf_token())


def n_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def target_token_len(target: str) -> int:
    # leading space matches the Stage-8 scoring convention (" Tamil" != "Tamil")
    return len(get_tokenizer().encode(" " + target, add_special_tokens=False))


def trailing_tokens(text: str, k: int) -> list[int]:
    return get_tokenizer().encode(text, add_special_tokens=False)[-k:]
