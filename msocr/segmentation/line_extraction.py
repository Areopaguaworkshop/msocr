"""Line extraction from serialized segmentation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image


def _bbox_from_line(line: dict) -> Tuple[int, int, int, int] | None:
    bbox = line.get("bbox")
    if bbox and len(bbox) == 4:
        return tuple(int(v) for v in bbox)
    boundary = line.get("boundary") or line.get("baseline")
    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return None


def extract_lines_from_segments(pages_dir: Path, segments_dir: Path, lines_dir: Path) -> int:
    lines_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for seg_path in sorted(segments_dir.glob("*.segments.json")):
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        image_path = Path(data["image"])
        if not image_path.exists():
            # Try resolve relative to pages_dir
            candidate = pages_dir / image_path.name
            if candidate.exists():
                image_path = candidate
            else:
                continue
        with Image.open(image_path) as img:
            for idx, line in enumerate(data.get("lines", []), start=1):
                bbox = _bbox_from_line(line)
                if not bbox:
                    continue
                crop = img.crop(bbox)
                out_name = f"{image_path.stem}_{idx:06d}.png"
                crop.save(lines_dir / out_name)
                count += 1
    return count
