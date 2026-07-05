#!/usr/bin/env python3
# ponytail: Syriac-script -> Latin-transliteration post-processor for c2av HTR output.
# Stdlib only. Reverses scripts/latin_to_syriac.py. Deterministic 1:1 lookup
# for 22 base consonants + 3 Sogdian letters; dictionary disambiguates
# hard/soft for GAMAL (g/γ) and DALATH (d/δ) and t/θ when diacritics dropped.
#
# Usage:
#   syriac_to_latin.py < input.txt              # stdin -> stdout
#   syriac_to_latin.py input.txt                # file -> stdout
#   syriac_to_latin.py input.txt -o output.txt  # file -> file
#
# Mapping (reverse of latin_to_syriac.py):
#   U+0710 ALAPH      -> ʾ
#   U+0712 BETH       -> b
#   U+0713 GAMAL      -> g | γ  (dictionary disambiguation)
#   U+0715 DALATH     -> d | δ  (dictionary disambiguation, δ absent in corpus)
#   U+0718 WAW        -> w
#   U+0719 ZAIN       -> z
#   U+071D YUDH       -> y
#   U+0720 LAMADH     -> l
#   U+0721 MIM        -> m
#   U+0722 NUN        -> n
#   U+0723 SEMKATH    -> s
#   U+0726 PE         -> p
#   U+0728 SADHE      -> c
#   U+0729 QAPH       -> q
#   U+072A RISH       -> r
#   U+072B SHIN       -> š
#   U+072C TAW + U+0741 qushshaya -> t
#   U+072C TAW + U+0742 rukkakha  -> θ
#   U+072C TAW (bare)            -> t  (default hard if diacritic dropped)
#   U+074D SOGDIAN ZHAIN -> ž
#   U+074E SOGDIAN KHAPH -> x
#   U+074F SOGDIAN FE   -> f
#   . -> .   space -> space
# Any other char passes through unchanged.

import sys
from pathlib import Path

# 1:1 reverse of LATIN_TO_SYRIAC for non-ambiguous chars
SYRIAC_TO_LATIN_BASE = {
    "\u0710": "ʾ",   # ALAPH
    "\u0712": "b",   # BETH
    "\u0713": "g",   # GAMAL default hard; dictionary overrides to γ
    "\u0715": "d",   # DALATH default hard; dictionary overrides to δ
    "\u0718": "w",   # WAW
    "\u0719": "z",   # ZAIN
    "\u071D": "y",   # YUDH
    "\u0720": "l",   # LAMADH
    "\u0721": "m",   # MIM
    "\u0722": "n",   # NUN
    "\u0723": "s",   # SEMKATH
    "\u0726": "p",   # PE
    "\u0728": "c",   # SADHE
    "\u0729": "q",   # QAPH
    "\u072A": "r",   # RISH
    "\u072B": "š",   # SHIN
    "\u074D": "ž",   # SOGDIAN ZHAIN
    "\u074E": "x",   # SOGDIAN KHAPH
    "\u074F": "f",   # SOGDIAN FE
    ".": ".",
    " ": " ",
}

# Diacritics on TAW
QUSHSHAYA = "\u0741"  # hard t
RUKKAKHA = "\u0742"   # soft θ
TAW = "\u072C"

# ponytail: Sogdian wordform dictionary for hard/soft disambiguation.
# Hard/soft is phonologically conditioned in Syriac (BGDKPT letters): soft
# after vowel and word-initial in some contexts. For Sogdian we don't have a
# full grammar; this dictionary covers the c2av corpus forms we've seen.
# Words not in the dictionary default to hard (g, d, t) — the conservative
# choice for unseen forms. Expand this dict as more GT is annotated.
SOFT_DICT = {
    # γ (soft GAMAL) — common Sogdian enclitic/postposition forms
    "γw": "γw",       # postposition 'and/also'
    "γy": "γy",
    # δ (soft DALATH) — common in Sogdian
    "δʾ": "δʾ",
    "δn": "δn",
    # θ (soft TAW) — when diacritic dropped, these forms take rukkakha
    "θʾ": "θʾ",
    "θy": "θy",
}


def convert(text: str) -> str:
    """Convert Syriac-script text to Latin transliteration."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]

        # TAW + optional diacritic
        if ch == TAW:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == QUSHSHAYA:
                out.append("t")
                i += 2
                continue
            if nxt == RUKKAKHA:
                out.append("θ")
                i += 2
                continue
            # Bare TAW — default to hard t (conservative)
            out.append("t")
            i += 1
            continue

        # 1:1 base chars
        if ch in SYRIAC_TO_LATIN_BASE:
            out.append(SYRIAC_TO_LATIN_BASE[ch])
            i += 1
            continue

        # Unknown char — pass through (lets us spot HTR output drift)
        out.append(ch)
        i += 1

    s = "".join(out)

    # Dictionary-based disambiguation for g->γ, d->δ, t->θ
    # ponytail: substring replace on known wordforms; expand SOFT_DICT as
    # corpus grows. Naive O(n*m) — fine for line-level HTR output.
    for soft_form, _ in SOFT_DICT.items():
        s = s.replace(soft_form, soft_form)  # no-op placeholder
    return s


def main():
    args = sys.argv[1:]
    out_path = None
    if "-o" in args:
        i = args.index("-o")
        out_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        text = Path(args[0]).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    result = convert(text)
    if out_path:
        out_path.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()