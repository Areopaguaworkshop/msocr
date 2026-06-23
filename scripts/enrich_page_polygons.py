#!/usr/bin/env python3
"""Enrich a PAGE XML by computing <Coords> polygons for each <TextLine>.

kraken 7.x requires both <Baseline> and <Coords> per <TextLine>. Our annotation
tool only exports <Baseline>. This script computes polygonal environments from
the baselines + source image and writes a new XML with <Coords> attached.

Usage:
    uv run python3 scripts/enrich_page_polygons.py INPUT.xml IMAGE.png OUTPUT.xml

Ponytail: standalone, no config, no class. One-shot per file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lxml.etree as ET
from PIL import Image

from kraken.lib.segmentation import calculate_polygonal_environment

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def enrich(src_xml: Path, image: Path, out_xml: Path) -> tuple[int, int]:
    """Return (n_lines, n_polys)."""
    tree = ET.parse(str(src_xml))
    root = tree.getroot()
    page = root.find(f"{{{NS}}}Page")
    if page is None:
        raise ValueError(f"No <Page> in {src_xml}")
    im = Image.open(str(image)).convert("L")

    baselines: list[list[tuple[int, int]]] = []
    lines_xml: list[ET._Element] = []
    for tl in page.iter(f"{{{NS}}}TextLine"):
        bl = tl.find(f"{{{NS}}}Baseline")
        if bl is None:
            continue
        pts = [(int(x), int(y)) for x, y in (p.split(",") for p in bl.get("points").split())]
        baselines.append(pts)
        lines_xml.append(tl)

    polys = calculate_polygonal_environment(im=im, baselines=baselines, topline=False, raise_on_error=False)
    ok = 0
    for tl, poly in zip(lines_xml, polys):
        if poly is None:
            continue
        pts_str = " ".join(f"{int(x)},{int(y)}" for x, y in poly)
        coords = ET.SubElement(tl, f"{{{NS}}}Coords")
        coords.set("points", pts_str)
        ok += 1
    tree.write(str(out_xml), xml_declaration=True, encoding="utf-8")
    return len(lines_xml), ok


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: enrich_page_polygons.py INPUT.xml IMAGE.png OUTPUT.xml")
    src, img, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n, k = enrich(src, img, out)
    print(f"{src}: {k}/{n} lines got polygons -> {out}")