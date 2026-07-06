#!/usr/bin/env python3
"""Compute CER between predicted and GT Latin transliteration.
Usage: cer.py pred.txt gt.txt
Ponytail: stdlib Levenshtein over character level (ignore whitespace? no — include spaces).
"""
from __future__ import annotations
import sys
from pathlib import Path


def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[lb]


def main() -> int:
    pred = Path(sys.argv[1]).read_text().splitlines()
    gt = Path(sys.argv[2]).read_text().splitlines()
    # ponytail: align by line index. If counts differ, trim to min.
    n = min(len(pred), len(gt))
    total_err = 0
    total_gt = 0
    for i in range(n):
        p, g = pred[i], gt[i]
        # normalize: strip, keep spaces as char
        err = levenshtein(p, g)
        total_err += err
        total_gt += len(g)
        print(f"L{i+1:2d} CER={100*err/max(len(g),1):.1f}% ({err}/{len(g)}) pred={p!r} gt={g!r}")
    cer = 100 * total_err / max(total_gt, 1)
    print(f"\n=== TOTAL CER: {cer:.2f}% ({total_err}/{total_gt}) over {n} lines ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())