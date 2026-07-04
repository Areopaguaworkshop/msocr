#!/usr/bin/env python3
"""3.1 Alphabet audit: dump converted GT alphabet, diff vs expected 25-char
list, check NFC/NFD, assess l singleton. Local, no GPU."""
import re, collections, unicodedata
from pathlib import Path

gt_dir = Path("dataset/christian_sogdian_c2av/gt")
txt = "".join(p.read_text(encoding="utf-8") for p in sorted(gt_dir.glob("*.xml")))
unicodes = re.findall(r"<Unicode>(.*?)</Unicode>", txt)
chars = collections.Counter()
for u in unicodes:
    for c in u:
        chars[c] += 1

print("=== Alphabet (NFC) ===")
alphabet = sorted(chars)
for c in alphabet:
    print(f"  U+{ord(c):04X} {unicodedata.name(c, '<unn>'):30s} x{chars[c]}")
print(f"\nTotal unique: {len(alphabet)}")

print("\n=== NFC/NFD check ===")
nfc = "".join(unicodedata.normalize("NFC", c) for c in alphabet)
nfd = "".join(unicodedata.normalize("NFD", c) for c in alphabet)
if nfc == "".join(alphabet):
    print("  Alphabet is in NFC. Good.")
else:
    print("  WARNING: alphabet not in NFC — may have normalization duplicates")
if nfd == "".join(alphabet):
    print("  Alphabet is in NFD.")
else:
    print("  Alphabet not in NFD (expected — Syriac combining marks usually precomposed)")

print("\n=== Combining mark analysis ===")
for c in alphabet:
    cat = unicodedata.category(c)
    if cat.startswith("M"):
        decomp = unicodedata.decomposition(c)
        print(f"  U+{ord(c):04X} {unicodedata.name(c)} cat={cat} x{chars[c]} decomp={decomp!r}")

print("\n=== l singleton (U+0720 LAMADH) ===")
print(f"  U+0720 LAMADH x{chars.get(chr(0x0720), 0)}")
print("  1 occurrence across 179 lines — cannot be learned. Decision needed.")

expected = {
    0x0710:"ALAPH", 0x0712:"BETH", 0x0713:"GAMAL", 0x0715:"DALATH",
    0x0718:"WAW", 0x0719:"ZAIN", 0x071D:"YUDH", 0x0720:"LAMADH",
    0x0721:"MIM", 0x0722:"NUN", 0x0723:"SEMKATH", 0x0725:"E",
    0x0726:"PE", 0x0728:"SADHE", 0x0729:"QOP", 0x072A:"RISH",
    0x072B:"SHIN", 0x072C:"TAW", 0x0741:"QUSHSHAYA", 0x0742:"RUKKAKHA",
    0x074D:"SOGDIAN ZHAIN", 0x074E:"SOGDIAN KHAPH", 0x074F:"SOGDIAN FE",
    0x0020:"SPACE", 0x002E:".",
}
print("\n=== Diff vs expected ===")
got = set(ord(c) for c in alphabet)
exp = set(expected)
if got == exp:
    print("  EXACT MATCH (25 chars incl space+dot)")
else:
    print(f"  In GT but not expected: {[hex(c) for c in got-exp]}")
    print(f"  Expected but not in GT: {[hex(c) for c in exp-got]}")