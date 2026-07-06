#!/usr/bin/env python3
"""Back-convert Sims-Williams caps-emphatic to our Latin transliteration.
Ponytail: reverse of LATIN_TO_SW. Lossy — SW 'G' could be γ or g (we collapsed),
so we pick the common case. Used only for measurement, not production.
"""
SW_TO_LATIN = {
    "A": "ʾ", "G": "γ", "T": "θ", "S": "š", "J": "ž", "X": "x",
    "W": "w", "Z": "z", "D": "d", "B": "b", "C": "c", "F": "f",
    "L": "l", "M": "m", "N": "n", "P": "p", "Q": "q", "R": "r",
    "Y": "y", "H": "h",
}

def back(text: str) -> str:
    return "".join(SW_TO_LATIN.get(c, c) for c in text)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    src = Path(sys.argv[1]).read_text()
    Path(sys.argv[2]).write_text(back(src))