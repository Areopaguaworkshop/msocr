"""Kraken BLLA segmentation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from kraken.blla import segment
from kraken.lib import models
from PIL import Image


def segment_pages(input_dir: Path, output_dir: Path, cfg: Dict) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    model_cfg = cfg.get("model")
    reading_order = str(cfg.get("reading_order", "rtl")).lower()
    text_direction = "horizontal-rl" if reading_order == "rtl" else "horizontal-lr"

    for image_path in sorted(input_dir.rglob("*.png")):
        with Image.open(image_path) as img:
            if not model_cfg or str(model_cfg).lower() == "blla":
                seg = segment(img, text_direction=text_direction)
            else:
                seg = segment(img, text_direction=text_direction, model=models.load_any(model_cfg))
        seg_json = _serialize_segmentation(seg, str(image_path))
        out_path = output_dir / f"{image_path.stem}.segments.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(seg_json, f, ensure_ascii=False, indent=2)
        count += 1
    return count


def _serialize_segmentation(seg, image_path: str) -> dict:
    lines = []
    for line in getattr(seg, "lines", []):
        bbox = getattr(line, "bbox", None)
        boundary = getattr(line, "boundary", None)
        baseline = getattr(line, "baseline", None)
        lines.append(
            {
                "id": getattr(line, "id", None),
                "bbox": list(bbox) if bbox else None,
                "boundary": [list(p) for p in boundary] if boundary else None,
                "baseline": [list(p) for p in baseline] if baseline else None,
            }
        )
    return {"image": image_path, "lines": lines}
