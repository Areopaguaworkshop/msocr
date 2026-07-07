#!/usr/bin/env python3
"""Produce a per-line contact sheet PNG with auto-polygons overlaid.

For QA of bootstrap polygons from enrich_page_polygons.py: renders each
TextLine crop with its <Coords> polygon drawn on top, so bad polygons
(lines that capture too much/little, mid-line collisions) are visible at
a glance. Lines are stacked vertically in reading order with a 20px gap.

Usage:
    uv run python3 scripts/qa_polygons.py INPUT.xml IMAGE.png OUTPUT.png

Ponytail: one-shot, no CLI framework, no config. PIL only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lxml.etree as ET
from PIL import Image, ImageDraw

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _pts(s: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in (p.split(",") for p in s.split())]


def render(src_xml: Path, image: Path, out_png: Path) -> int:
    tree = ET.parse(str(src_xml))
    page = tree.getroot().find(f"{{{NS}}}Page")
    im = Image.open(str(image)).convert("RGB")

    crops: list[tuple[Image.Image, list[tuple[int, int]], list[tuple[int, int]], str]] = []
    for i, tl in enumerate(page.iter(f"{{{NS}}}TextLine")):
        bl = tl.find(f"{{{NS}}}Baseline")
        co = tl.find(f"{{{NS}}}Coords")
        te = tl.find(f"{{{NS}}}TextEquiv")
        if bl is None or co is None:
            continue
        poly = _pts(co.get("points"))
        xs = [x for x, _ in poly]; ys = [y for _, y in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad = 10
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(im.width, x1 + pad); y1 = min(im.height, y1 + pad)
        crop = im.crop((x0, y0, x1, y1))
        local_poly = [(x - x0, y - y0) for x, y in poly]
        local_bl = [(x - x0, y - y0) for x, y in _pts(bl.get("points"))]
        txt = ""
        if te is not None:
            u = te.find(f"{{{NS}}}Unicode")
            txt = (u.text or "") if u is not None else ""
        crops.append((crop, local_poly, local_bl, f"L{i}: {txt[:40]}"))

    if not crops:
        out_png.write_bytes(b""); return 0

    max_w = max(c.width for c, *_ in crops)
    total_h = sum(c.height for c, *_ in crops) + 20 * (len(crops) - 1) + 30 * len(crops)
    sheet = Image.new("RGB", (max_w + 20, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    y = 0
    for crop, poly, bl, label in crops:
        sheet.paste(crop, (10, y + 30))
        d = ImageDraw.Draw(sheet)
        d.line([(p[0] + 10, p[1] + y + 30) for p in poly] + [(poly[0][0] + 10, poly[0][1] + y + 30)],
               fill=(0, 255, 0), width=2)
        d.line([(p[0] + 10, p[1] + y + 30) for p in bl], fill=(255, 0, 255), width=2)
        draw.text((10, y + 5), label, fill=(255, 255, 255))
        y += crop.height + 30 + 20

    sheet.save(str(out_png))
    return len(crops)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: qa_polygons.py INPUT.xml IMAGE.png OUTPUT.png")
    n = render(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"{n} lines -> {sys.argv[3]}")