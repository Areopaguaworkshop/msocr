"""Text evaluation metrics for OCR benchmarking."""

from __future__ import annotations

from typing import List


def _levenshtein(seq_a: List[str], seq_b: List[str]) -> int:
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, token_b in enumerate(seq_b, start=1):
            ins = prev[j] + 1
            delete = curr[j - 1] + 1
            sub = prev[j - 1] + (token_a != token_b)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate."""
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if not ref_chars:
        return 1.0 if hyp_chars else 0.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)
