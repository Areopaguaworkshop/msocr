"""Mixed-script token-level language correction helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"(\s+|[^\w\u0700-\u074F]+)")


def _load_lexicon(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    words: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if token:
            words.add(token)
    return words


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_syriac(token: str, syriac_range: tuple[int, int]) -> bool:
    start, end = syriac_range
    return any(start <= ord(ch) <= end for ch in token)


def _closest(token: str, lexicon: set[str], max_edit_distance: int) -> str:
    if not lexicon or token in lexicon:
        return token
    best = token
    best_dist = max_edit_distance + 1
    token_head = token[:1]
    for candidate in lexicon:
        if token_head and candidate[:1] != token_head:
            continue
        dist = _levenshtein(token, candidate)
        if dist < best_dist:
            best = candidate
            best_dist = dist
            if dist == 1:
                break
    return best if best_dist <= max_edit_distance else token


def _correct_text(
    text: str,
    *,
    syriac_lexicon: set[str],
    latin_lexicon: set[str],
    max_edit_distance: int,
    syriac_range: tuple[int, int],
) -> str:
    out: list[str] = []
    for token in TOKEN_RE.split(text):
        if not token or TOKEN_RE.fullmatch(token):
            out.append(token)
            continue
        if _is_syriac(token, syriac_range):
            out.append(_closest(token, syriac_lexicon, max_edit_distance))
        elif token.isascii() and token.isalpha():
            out.append(_closest(token.lower(), latin_lexicon, max_edit_distance))
        else:
            out.append(token)
    return "".join(out)


def _iter_json_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*.json")):
        if path.is_file():
            yield path


def correct_ocr_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    syriac_lexicon_path: Path | None,
    latin_lexicon_path: Path | None,
    max_edit_distance: int = 1,
    syriac_range: tuple[int, int] = (0x0700, 0x074F),
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    syriac_lexicon = _load_lexicon(syriac_lexicon_path)
    latin_lexicon = _load_lexicon(latin_lexicon_path)
    corrected = 0

    for json_file in _iter_json_files(input_dir):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        text = (
            payload.get("transcription")
            or payload.get("text")
            or payload.get("full_text")
            or ""
        )
        payload["corrected_text"] = _correct_text(
            str(text),
            syriac_lexicon=syriac_lexicon,
            latin_lexicon=latin_lexicon,
            max_edit_distance=max_edit_distance,
            syriac_range=syriac_range,
        )
        out_path = output_dir / json_file.name
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        corrected += 1

    return corrected
