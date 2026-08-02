"""Line extraction from serialized segmentation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


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


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_fraction: float,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    line_height = max(bottom - top, 1)
    pad_x = max(2, round(line_height * padding_fraction))
    pad_y = max(2, round(line_height * padding_fraction / 2))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x + 1),
        min(height, bottom + pad_y + 1),
    )


def _write_contact_sheet(lines_dir: Path, output_path: Path) -> None:
    crops = sorted(lines_dir.glob("*.png"))
    if not crops:
        return
    width = 900
    pad = 12
    label_height = 22
    font = ImageFont.load_default()
    rows: list[Image.Image] = []
    for path in crops:
        with Image.open(path) as source:
            image = source.convert("RGB")
        scale = min(width / image.width, 160 / image.height)
        thumbnail = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        )
        row = Image.new("RGB", (width, label_height + thumbnail.height), "white")
        ImageDraw.Draw(row).text((4, 4), path.name, fill="black", font=font)
        row.paste(thumbnail, (0, label_height))
        rows.append(row)
    sheet = Image.new(
        "RGB", (width + pad * 2, sum(row.height + pad for row in rows) + pad), "#eeeeee"
    )
    y = pad
    for row in rows:
        sheet.paste(row, (pad, y))
        y += row.height + pad
    sheet.save(output_path, quality=90)


def extract_lines_from_segments(
    pages_dir: Path,
    segments_dir: Path,
    lines_dir: Path,
    *,
    padding_fraction: float = 0.25,
    provenance_path: Path | None = None,
    contact_sheet_path: Path | None = None,
) -> int:
    """Crop BLLA proposals as atomic, review-required line-fragment images."""
    lines_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    provenance: list[dict] = []
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
                bbox = _pad_bbox(bbox, img.size, padding_fraction)
                crop = img.crop(bbox)
                out_name = f"{image_path.stem}_{idx:06d}.png"
                crop.save(lines_dir / out_name)
                provenance.append(
                    {
                        "proposal_id": line.get("id") or f"{image_path.stem}_{idx:06d}",
                        "source_fragment": str(image_path),
                        "crop": str(lines_dir / out_name),
                        "bbox": list(bbox),
                        "baseline": line.get("baseline") or [],
                        "boundary": line.get("boundary") or [],
                        "status": "proposal_only",
                        "training_eligible": False,
                        "review_required": True,
                    }
                )
                count += 1
    if provenance_path is not None:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps({"proposals": provenance}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if contact_sheet_path is not None:
        _write_contact_sheet(lines_dir, contact_sheet_path)
    return count
