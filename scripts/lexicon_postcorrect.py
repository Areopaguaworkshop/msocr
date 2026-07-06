#!/usr/bin/env python3
"""3.1 Lexicon post-corrector — extract Sogdian wordlist from Sims-Williams
2021 Christian Sogdian dictionary corpus, then edit-distance-correct HTR output.

Pipeline: HTR (Syriac) → syriac_to_latin.py → normalize to SW caps-emphatic
→ lexicon post-correct → final Latin.

The lexicon uses Sims-Williams' caps-for-emphatic ASCII convention:
  A=ʾ (aleph)   G=γ (gamal)  J=ž (zhain)
  S=š (shin)    T=θ (taw)    X=x (khaph)
  W=w           Z=z          D=d (dalath, emphatic d)
  C=c?          d=d          l=l
Our syriac_to_latin emits lowercase with diacritics (γ θ š ž ʾ). We normalize
to SW convention before matching, then de-normalize back.

Two modes:
  --build-lexicon  : parse NSW2021corpus.xls col 2, write dataset/lexicon/sogdian_words.txt
  --correct <file> : read Latin transliteration lines, correct each token against lexicon

Ponytail: stdlib only for the corrector. xlrd for lexicon build (one-time).
Edit distance is Levenshtein ≤ 2, prefer fewest edits, only substitute if HTR
token is NOT already in lexicon (don't touch attested words).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON_XLS = ROOT / "dataset/lexicon/NSW2021corpus.xls"
LEXICON_TXT = ROOT / "dataset/lexicon/sogdian_words.txt"

# Our Latin → Sims-Williams caps-emphatic. Ponytail: lossy on purpose — the
# lexicon doesn't preserve all distinctions, so we match in SW space and
# return SW; the caller can decide whether to back-convert.
LATIN_TO_SW = str.maketrans({
    "ʾ": "A", "ʼ": "A", "'": "A",
    "γ": "G",
    "θ": "T",
    "š": "S",
    "ž": "J",
    "x": "X",
    "w": "W",
    "z": "Z",
    "d": "D",
    "b": "B", "c": "C", "f": "F", "g": "G",  # g maps to G too (gamal collapse)
    "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
    "s": "S", "t": "T",
    "y": "Y",
    "h": "H",
})


def to_sw(text: str) -> str:
    """Normalize our Latin transliteration to Sims-Williams caps-emphatic."""
    return text.translate(LATIN_TO_SW).upper()


def build_lexicon() -> int:
    """Extract unique words from Sims-Williams corpus col 2 ('Sogdian searchable')."""
    import xlrd
    wb = xlrd.open_workbook(str(LEXICON_XLS))
    sh = wb.sheet_by_index(0)
    words: set[str] = set()
    # ponytail: col 2 'Sogdian searchable' uses caps-for-emphatic ASCII convention.
    token_re = re.compile(r"[A-Za-z]+(?:[-'ʼ][A-Za-z]+)*")
    for r in range(1, sh.nrows):
        val = str(sh.cell_value(r, 2)).strip()
        if not val or val == "…" or val == "...":
            continue
        for tok in token_re.findall(val):
            tok = tok.strip(".-")
            if len(tok) >= 2:
                words.add(tok)
    words_sorted = sorted(words)
    LEXICON_TXT.parent.mkdir(parents=True, exist_ok=True)
    LEXICON_TXT.write_text("\n".join(words_sorted) + "\n")
    print(f"wrote {LEXICON_TXT}: {len(words_sorted)} unique words", file=sys.stderr)
    return 0


def load_lexicon() -> set[str]:
    if not LEXICON_TXT.exists():
        print(f"missing lexicon: {LEXICON_TXT} — run with --build-lexicon first", file=sys.stderr)
        sys.exit(1)
    return set(w.strip() for w in LEXICON_TXT.read_text().splitlines() if w.strip())


def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def correct_token(tok_sw: str, lexicon: set[str], max_dist: int = 2) -> tuple[str, str | None]:
    """Return (final_token, correction_or_None). Only correct if tok not in lexicon."""
    if tok_sw in lexicon:
        return tok_sw, None
    if len(tok_sw) < 2:
        return tok_sw, None
    best: str | None = None
    best_dist = max_dist + 1
    for w in lexicon:
        if abs(len(w) - len(tok_sw)) > max_dist:
            continue
        d = _levenshtein(tok_sw, w, max_dist)
        if d < best_dist:
            best_dist = d
            best = w
            if d == 1:
                break
    if best is not None and best_dist <= max_dist:
        return best, f"{tok_sw}→{best}(d={best_dist})"
    return tok_sw, None


def correct_lines(lines: list[str], lexicon: set[str], max_dist: int = 2) -> tuple[list[str], list[str]]:
    """Correct each whitespace-tokenized word. Returns (corrected_lines, corrections_log)."""
    out: list[str] = []
    log: list[str] = []
    for i, line in enumerate(lines, 1):
        tokens = line.split()
        corrected: list[str] = []
        for tok in tokens:
            bare = tok.strip(".,;:!?")
            punct_pre = tok[:len(tok) - len(tok.lstrip(".,;:!?"))]
            punct_post = tok[len(tok.rstrip(".,;:!?")):]
            tok_sw = to_sw(bare)
            final, corr = correct_token(tok_sw, lexicon, max_dist)
            corrected.append(punct_pre + final + punct_post)
            if corr:
                log.append(f"L{i}: {corr}")
        out.append(" ".join(corrected))
    return out, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-lexicon", action="store_true")
    ap.add_argument("--correct", metavar="FILE", help="correct Latin transliteration file")
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("-o", "--output", metavar="FILE", help="write corrected output here")
    args = ap.parse_args()

    if args.build_lexicon:
        return build_lexicon()
    if args.correct:
        lex = load_lexicon()
        src = Path(args.correct).read_text().splitlines()
        corrected, log = correct_lines(src, lex, args.max_dist)
        out_path = Path(args.output) if args.output else Path(args.correct).with_suffix(".corr.txt")
        out_path.write_text("\n".join(corrected) + "\n")
        log_path = out_path.with_suffix(".log.txt")
        log_path.write_text("\n".join(log) + ("\n" if log else ""))
        print(f"wrote {out_path} ({len(corrected)} lines)", file=sys.stderr)
        print(f"wrote {log_path} ({len(log)} corrections)", file=sys.stderr)
        total_tokens = sum(len(l.split()) for l in src)
        print(f"\n=== lexicon post-correction ===")
        print(f"input tokens: {total_tokens}")
        print(f"corrections:  {len(log)} ({100*len(log)/max(total_tokens,1):.1f}% of tokens)")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())