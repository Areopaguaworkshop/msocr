"""Synthetic Syriac line generation for Payne-Smith domain adaptation."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_WORDS = [
    "ܐܒܐ",
    "ܡܠܬܐ",
    "ܟܬܒܐ",
    "ܣܘܪܝܝܐ",
    "ܕܝܠܢ",
    "ܪܒܐ",
    "ܡܕܪܫܐ",
    "ܦܪܘܫܐ",
    "ܣܦܪܐ",
    "ܥܒܕܐ",
]


def _iter_font_paths() -> Iterable[Path]:
    roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.ttf", "*.otf"):
            yield from root.rglob(ext)


def _resolve_font(candidates: Sequence[str], size: int) -> ImageFont.ImageFont:
    if candidates:
        lower_candidates = [c.lower() for c in candidates]
        for font_path in _iter_font_paths():
            name = font_path.name.lower()
            if any(c in name for c in lower_candidates):
                try:
                    return ImageFont.truetype(str(font_path), size=size)
                except OSError:
                    continue
    return ImageFont.load_default()


def _build_line(words: Sequence[str], rng: random.Random) -> str:
    length = rng.randint(2, 6)
    return " ".join(rng.choice(words) for _ in range(length))


def _write_page_xml(xml_path: Path, image_name: str, width: int, height: int, text: str) -> None:
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    ET.register_namespace("", ns)
    pcgts = ET.Element(f"{{{ns}}}PcGts")
    page = ET.SubElement(
        pcgts,
        f"{{{ns}}}Page",
        {
            "imageFilename": image_name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )
    region = ET.SubElement(page, f"{{{ns}}}TextRegion", {"id": "r1"})
    ET.SubElement(region, f"{{{ns}}}Coords", {"points": f"0,0 {width},0 {width},{height} 0,{height}"})
    line = ET.SubElement(region, f"{{{ns}}}TextLine", {"id": "l1"})
    ET.SubElement(line, f"{{{ns}}}Baseline", {"points": f"12,{height - 8} {width - 12},{height - 8}"})
    ET.SubElement(line, f"{{{ns}}}Coords", {"points": f"8,8 {width - 8},8 {width - 8},{height - 8} 8,{height - 8}"})
    text_equiv = ET.SubElement(line, f"{{{ns}}}TextEquiv")
    unicode_el = ET.SubElement(text_equiv, f"{{{ns}}}Unicode")
    unicode_el.text = text
    tree = ET.ElementTree(pcgts)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def generate_synthetic_lines(
    *,
    output_dir: Path,
    count: int,
    line_height: int = 48,
    seed: int = 42,
    fonts: Sequence[str] | None = None,
    words: Sequence[str] | None = None,
    augment: bool = True,
) -> int:
    """Generate synthetic line-image + PAGE-XML pairs.

    This is a lightweight fallback used when TRDG or external scripts are unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    lexicon = list(words) if words else list(DEFAULT_WORDS)
    font = _resolve_font(fonts or [], size=max(20, int(line_height * 0.7)))
    generated = 0

    for idx in range(1, count + 1):
        text = _build_line(lexicon, rng)
        width = max(320, int(len(text) * line_height * 0.9))
        height = max(48, line_height + 16)
        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)
        draw.text((12, 8), text, fill=0, font=font)

        if augment:
            if rng.random() < 0.35:
                image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
            if rng.random() < 0.4:
                arr = np.array(image)
                noise = rng.randint(3, 9)
                arr = np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
                image = Image.fromarray(arr, mode="L")

        stem = f"synthetic_{idx:06d}"
        image_name = f"{stem}.png"
        xml_name = f"{stem}.xml"
        image.save(output_dir / image_name)
        _write_page_xml(output_dir / xml_name, image_name, width, height, text)
        generated += 1

    return generated
