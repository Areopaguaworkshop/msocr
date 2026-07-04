#!/usr/bin/env python3
# ponytail: Latin-transliteration -> Syriac-script converter for c2av PAGE-XML.
# Stdlib only. In-place with .bak backups. Emits char-frequency report.
#
# Mapping decisions (user-confirmed 2026-07-04):
#   t/θ  -> TAW (U+072C) + diacritic: t=qushshaya U+0741, θ=rukkakha U+0742
#   x    -> KHAPH U+074E (Sogdian-specific [x])
#   g/γ  -> GAMAL U+0713 (collapse hard/soft)
#   d/δ  -> DALATH U+0715 (collapse; δ absent in corpus)
#   l    -> LAMADH U+0720 (keep, 1 occurrence)
#   ž    -> SOGDIAN ZHAIN U+074D
#   f    -> SOGDIAN FE U+074F
#   š    -> SHIN U+072B
#   ʾ    -> ALAPH U+0710
#   All other 24-char inventory -> standard Syriac per Unicode block
#
# New classes added via --resize add: U+074D, U+074E, U+074F, U+0741, U+0742 (5 total)

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from collections import Counter

NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}"
ET.register_namespace("", NS[1:-1])

# Latin char -> Syriac string (string because t/θ emit 2 codepoints)
LATIN_TO_SYRIAC = {
    "ʾ": "\u0710",   # ALAPH
    "b": "\u0712",   # BETH
    "g": "\u0713",   # GAMAL (collapsed with γ)
    "γ": "\u0713",   # GAMAL (collapsed)
    "d": "\u0715",   # DALATH (collapsed with δ, which is absent)
    "w": "\u0718",   # WAW
    "z": "\u0719",   # ZAIN
    "y": "\u071D",   # YUDH
    "x": "\u074E",   # SOGDIAN KHAPH (NEW)
    "l": "\u0720",   # LAMADH
    "m": "\u0721",   # MIM
    "n": "\u0722",   # NUN
    "s": "\u0723",   # SEMKATH
    "p": "\u0726",   # PE
    "c": "\u0728",   # SADHE
    "q": "\u0729",   # QAPH
    "r": "\u072A",   # RISH
    "š": "\u072B",   # SHIN
    "t": "\u072C\u0741",  # TAW + qushshaya (hard)
    "θ": "\u072C\u0742",  # TAW + rukkakha (soft)
    "ž": "\u074D",   # SOGDIAN ZHAIN (NEW)
    "f": "\u074F",   # SOGDIAN FE (NEW)
    ".": ".",        # sentence punctuation (U+002E, in base)
    " ": " ",        # word separator (U+0020, in base)
}

# Syriac chars that are already valid output and should pass through untouched
# (for c2av19 line `ܫܪܥܦ` etc.)
SYRIAC_PASS_THROUGH = set()
for cp in range(0x0700, 0x0750):
    SYRIAC_PASS_THROUGH.add(chr(cp))

# New codepoints not in the base codec (need --resize add)
NEW_CODEPOINTS = {"\u074D", "\u074E", "\u074F", "\u0741", "\u0742"}


def convert_line(text: str) -> tuple[str, list[str]]:
    """Convert Latin transliteration to Syriac script.
    Returns (converted_text, list_of_unmapped_chars).
    """
    out = []
    unmapped = []
    for ch in text:
        if ch in LATIN_TO_SYRIAC:
            out.append(LATIN_TO_SYRIAC[ch])
        elif ch in SYRIAC_PASS_THROUGH:
            # Already-Syriac chars (e.g. c2av19 `ܫܪܥܦ`) pass through
            out.append(ch)
        else:
            unmapped.append(ch)
            # Keep the unmapped char as-is so we can spot it in the report
            out.append(ch)
    return "".join(out), unmapped


def convert_xml(xml_path: Path, backup: bool = True) -> tuple[int, Counter, list[str]]:
    """Convert all <Unicode> text in a PAGE-XML. Returns (n_lines, char_counts, unmapped)."""
    if backup:
        bak = xml_path.with_suffix(xml_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(xml_path, bak)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    n_lines = 0
    chars = Counter()
    all_unmapped = []

    for line in root.iter(f"{NS}TextLine"):
        te = line.find(f"{NS}TextEquiv")
        if te is None:
            continue
        u = te.find(f"{NS}Unicode")
        if u is None or not u.text:
            continue
        converted, unmapped = convert_line(u.text)
        u.text = converted
        n_lines += 1
        chars.update(converted)
        all_unmapped.extend(unmapped)

    tree.write(xml_path, xml_declaration=True, encoding="utf-8")
    return n_lines, chars, all_unmapped


def main():
    gt_dir = Path("dataset/christian_sogdian_c2av/gt")
    xmls = sorted(gt_dir.glob("c2av*_page_*.xml"))
    if not xmls:
        raise SystemExit(f"No XMLs found in {gt_dir}")

    total_lines = 0
    total_chars = Counter()
    total_unmapped = []
    per_file = {}

    for xml in xmls:
        n, chars, unmapped = convert_xml(xml)
        total_lines += n
        total_chars.update(chars)
        total_unmapped.extend(unmapped)
        per_file[xml.stem] = (n, dict(chars), unmapped)
        print(f"  {xml.stem}: {n} lines, {len(chars)} unique chars, {len(unmapped)} unmapped")

    print()
    print("=== Total converted char counts across 12 plates ===")
    for ch, n in sorted(total_chars.items(), key=lambda x: -x[1]):
        cp = ord(ch) if ch else 0
        name = repr(ch) if ch.strip() or ch == " " else f"<U+{cp:04X}>"
        new_tag = " NEW" if ch in NEW_CODEPOINTS else ""
        print(f"  {name} U+{cp:04X}: {n}{new_tag}")

    print()
    print(f"Total lines: {total_lines}")
    print(f"Total chars (excl space): {sum(n for ch,n in total_chars.items() if ch.strip())}")
    print(f"Unique chars in output: {len(total_chars)}")
    print(f"NEW codepoints (need --resize add): {sorted(NEW_CODEPOINTS)}")
    for cp_char in sorted(NEW_CODEPOINTS):
        cp = ord(cp_char)
        print(f"  U+{cp:04X} {cp_char!r}: {total_chars.get(cp_char, 0)} occurrences")

    if total_unmapped:
        print()
        print(f"WARNING: {len(total_unmapped)} unmapped chars (kept as-is):")
        um_counts = Counter(total_unmapped)
        for ch, n in sorted(um_counts.items(), key=lambda x: -x[1]):
            cp = ord(ch) if ch else 0
            print(f"  {ch!r} U+{cp:04X}: {n}")
    else:
        print()
        print("All chars mapped successfully (no unmapped).")


if __name__ == "__main__":
    main()