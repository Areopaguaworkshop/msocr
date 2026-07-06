#!/usr/bin/env python3
"""Phase 0a — Dump the c2av union codec (base + 5 new Sogdian classes).

Loads the Sophro Mhiro Syriac base model, inspects `model.codec.c2l` for the
37-row base mapping, then computes the 42-row union map (base + 5 new
Sogdian classes added by `--resize union`).

The 5 new classes are the ones missing from the Sophro base and present in
the c2av GT after `scripts/latin_to_syriac.py` conversion. Their codepoints
are the highest in the Syriac block, so Kraken's `PytorchCodec` (which sorts
by codepoint) appends them as the last 5 rows.

Outputs:
  reports/c2av_union_codec.json  — machine-readable: old_rows, new_rows,
                                   char_to_row, row_to_char
  reports/c2av_union_codec.md    — human-readable table

This is the source of truth for which classifier rows to freeze in Phase 1
(the §1 classifier-row freeze fix from docs/fixa-v4-impl-plan-2026-07-06.md).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kraken.lib.vgsl import TorchVGSLModel

BASE_MODEL = ROOT / "models/kraken/sophro_mhiro_syriac.mlmodel"
REPORTS = ROOT / "reports"

# The 5 new Sogdian classes added by --resize union (from
# scripts/confusion_analyze.py:27 and docs/fix-a-v2-syriac-script-finetune.md:61-63).
# Names from Unicode U+0700 chart.
NEW_CLASS_CODEPOINTS = {
    0x0741: "SYRIAC QUSHSHAYA",
    0x0742: "SYRIAC RUKKAKHA",
    0x074D: "SYRIAC LETTER SOGDIAN ZHAIN",
    0x074E: "SYRIAC LETTER SOGDIAN KHAPH",
    0x074F: "SYRIAC LETTER SOGDIAN FE",
}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    model = TorchVGSLModel.load_model(str(BASE_MODEL))
    base_c2l: dict[str, list[int]] = dict(model.codec.c2l)

    # Base rows: row 0 is CTC blank (implicit, not in c2l). Rows 1..N are chars.
    # c2l values are lists; each char maps to one row here (no multi-codepoint
    # grapheme clusters in this codec).
    base_char_to_row = {ch: rows[0] for ch, rows in base_c2l.items() if rows}
    # row 0 = blank
    row_to_char = {0: "<BLANK>"}
    for ch, r in base_char_to_row.items():
        row_to_char[r] = ch

    base_max_row = max(row_to_char)
    # Sanity: weight shape is [num_classes, hidden]; num_classes = base_max_row+1
    lin = model.nn[-1]
    n_base_classes = lin.lin.weight.shape[0]
    if n_base_classes != base_max_row + 1:
        print(f"WARN: codec max row {base_max_row} != weight rows {n_base_classes}",
              file=sys.stderr)

    # Union: 5 new classes appended after base rows, sorted by codepoint
    # (Kraken's PytorchCodec sorts by codepoint). All 5 are > any base codepoint
    # in the Syriac block, so they land at the end.
    new_chars = sorted(NEW_CLASS_CODEPOINTS, key=lambda cp: cp)
    new_rows: list[dict] = []
    next_row = n_base_classes
    char_to_row_union = dict(base_char_to_row)
    row_to_char_union = dict(row_to_char)
    for cp in new_chars:
        ch = chr(cp)
        row = next_row
        new_rows.append({
            "row": row,
            "char": ch,
            "codepoint": f"U+{cp:04X}",
            "name": NEW_CLASS_CODEPOINTS[cp],
        })
        char_to_row_union[ch] = row
        row_to_char_union[row] = ch
        next_row += 1

    old_rows = [r for r in row_to_char if r not in {nr["row"] for nr in new_rows}]
    n_union = next_row

    # Cross-check: none of the 5 new codepoints should already be in base
    overlap = [cp for cp in NEW_CLASS_CODEPOINTS if chr(cp) in base_char_to_row]
    if overlap:
        print(f"WARN: codepoints already in base codec: {[f'U+{cp:04X}' for cp in overlap]}",
              file=sys.stderr)

    out_json = {
        "base_model": str(BASE_MODEL),
        "n_base_classes": n_base_classes,
        "n_union_classes": n_union,
        "ctc_blank_row": 0,
        "old_rows": sorted(old_rows),
        "new_rows": [nr["row"] for nr in new_rows],
        "new_class_details": new_rows,
        "char_to_row": {ch: r for ch, r in char_to_row_union.items()},
        "row_to_char": {str(r): ch for r, ch in row_to_char_union.items()},
    }
    (REPORTS / "c2av_union_codec.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2) + "\n")

    # Human-readable table
    lines = [
        "# c2av union codec (base + 5 new Sogdian classes)",
        "",
        f"Base model: `{BASE_MODEL}`",
        f"Base classes: **{n_base_classes}** (row 0 = CTC blank, rows 1..{n_base_classes-1} = 36 chars)",
        f"Union classes: **{n_union}** (5 new rows appended)",
        "",
        "## New classes (rows to LEAVE TRAINABLE in Phase 1)",
        "",
        "| Row | Char | Codepoint | Name |",
        "|---|---|---|---|",
    ]
    for nr in new_rows:
        lines.append(f"| {nr['row']} | {nr['char']} | {nr['codepoint']} | {nr['name']} |")
    lines += [
        "",
        f"## Old classes (rows to FREEZE in Phase 1): {len(old_rows)} rows",
        "",
        "| Row | Char |",
        "|---|---|",
    ]
    for r in sorted(old_rows):
        ch = row_to_char.get(r, "?")
        # row 0 is "<BLANK>" sentinel; only format single-char strings
        if len(ch) == 1:
            disp = repr(ch) if ch in (" ",) or ord(ch) < 0x20 else ch
        else:
            disp = ch
        lines.append(f"| {r} | {disp} |")
    lines += [
        "",
        "## Source / cross-check",
        "",
        "5 new classes match `NEW_CLASSES` in `scripts/confusion_analyze.py:27`:",
        "`{ݎ, ݏ, ݍ, SYRIAC QUSHSHAYA, SYRIAC RUKKAKHA}`.",
        "",
        "Row 0 is the CTC blank (always present, never trained).",
    ]
    (REPORTS / "c2av_union_codec.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {REPORTS / 'c2av_union_codec.json'}")
    print(f"Wrote {REPORTS / 'c2av_union_codec.md'}")
    print(f"Base: {n_base_classes} classes | Union: {n_union} classes")
    print(f"New rows (trainable in Phase 1): {[nr['row'] for nr in new_rows]}")
    print(f"Old rows (frozen in Phase 1): {len(old_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())