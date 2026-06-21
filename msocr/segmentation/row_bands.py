"""Exact-count row-band extraction for fragmented manuscript lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class LineBand:
    line_id: str
    order: int
    bbox: tuple[int, int, int, int]
    image_path: Path


def extract_row_bands(
    image_path: Path,
    output_dir: Path,
    *,
    expected_lines: int,
    roi: tuple[int, int, int, int] | None = None,
    row_centers: list[int] | None = None,
    min_component_area: int = 20,
) -> list[LineBand]:
    """Extract exactly expected_lines line-band crops from fragmented manuscript ink."""
    if expected_lines < 1:
        raise ValueError("expected_lines must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "lines"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("line_*.jpg"):
        old.unlink()

    image = Image.open(image_path).convert("RGB")
    left, top, right, bottom = _clamp_roi(roi, image.size)
    roi_image = image.crop((left, top, right, bottom))
    components = _ink_components(roi_image, min_component_area)
    if len(components) < expected_lines:
        raise ValueError(
            f"Only {len(components)} ink components found for {expected_lines} expected lines"
        )

    if row_centers is not None:
        if len(row_centers) != expected_lines:
            raise ValueError("row_centers length must match expected_lines")
        clusters = _components_by_row_centers(components, [center - top for center in row_centers])
    else:
        clusters = _cluster_components_by_y(components, expected_lines)
    bands: list[LineBand] = []
    for order, cluster in enumerate(clusters, start=1):
        x0 = min(c[0] for c in cluster) + left
        y0 = min(c[1] for c in cluster) + top
        x1 = max(c[2] for c in cluster) + left
        y1 = max(c[3] for c in cluster) + top
        bbox = _padded_bbox((x0, y0, x1, y1), image.size)
        line_id = f"line_{order:03d}"
        crop_path = crops_dir / f"{line_id}.jpg"
        image.crop(bbox).save(crop_path, format="JPEG", quality=95)
        bands.append(LineBand(line_id=line_id, order=order, bbox=bbox, image_path=crop_path))

    _write_overlay(image, bands, output_dir / "line_overlay.jpg")
    _write_contact_sheet(bands, output_dir / "line_contact_sheet.jpg")
    return bands


def _components_by_row_centers(
    components: list[tuple[int, int, int, int, int]], row_centers: list[int]
) -> list[list[tuple[int, int, int, int, int]]]:
    centers = sorted(int(center) for center in row_centers)
    cuts = [0]
    cuts.extend((a + b) // 2 for a, b in zip(centers, centers[1:]))
    cuts.append(10**9)

    clusters = []
    for top, bottom in zip(cuts, cuts[1:]):
        cluster = [
            component
            for component in components
            if top <= (component[1] + component[3]) / 2 < bottom
        ]
        if not cluster:
            raise ValueError(f"No ink components found for row band {top}..{bottom}")
        clusters.append(cluster)
    return clusters


def _clamp_roi(
    roi: tuple[int, int, int, int] | None, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    if roi is None:
        return 0, 0, width, height
    left, top, right, bottom = roi
    left = min(max(int(left), 0), width - 1)
    top = min(max(int(top), 0), height - 1)
    right = min(max(int(right), left + 1), width)
    bottom = min(max(int(bottom), top + 1), height)
    return left, top, right, bottom


def _ink_components(image: Image.Image, min_area: int) -> list[tuple[int, int, int, int, int]]:
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= min_area:
            components.append((x, y, x + w, y + h, area))
    return components


def _cluster_components_by_y(
    components: list[tuple[int, int, int, int, int]], expected_lines: int
) -> list[list[tuple[int, int, int, int, int]]]:
    centers = np.array([[(c[1] + c[3]) / 2] for c in components], dtype=np.float32)
    if len(components) == expected_lines:
        labels = np.arange(expected_lines)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(
            centers,
            expected_lines,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.ravel()

    clusters = []
    for label in sorted(set(int(v) for v in labels), key=lambda n: _cluster_y(components, labels, n)):
        cluster = [component for component, item_label in zip(components, labels) if int(item_label) == label]
        clusters.append(cluster)
    if len(clusters) != expected_lines:
        raise ValueError(f"Could not form exactly {expected_lines} line clusters")
    return clusters


def _cluster_y(
    components: list[tuple[int, int, int, int, int]], labels: Iterable[int], label: int
) -> float:
    ys = [(c[1] + c[3]) / 2 for c, item_label in zip(components, labels) if int(item_label) == label]
    return float(np.mean(ys))


def _padded_bbox(
    bbox: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = bbox
    line_height = max(bottom - top, 1)
    pad_x = max(30, line_height)
    pad_y = max(10, line_height // 3)
    return _clamp_roi((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), size)


def _write_overlay(image: Image.Image, bands: list[LineBand], path: Path) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for band in bands:
        draw.rectangle(band.bbox, outline=(220, 30, 30), width=3)
        draw.text((band.bbox[0] + 4, band.bbox[1] + 4), str(band.order), fill=(220, 30, 30), font=font)
    overlay.save(path, quality=90)


def _write_contact_sheet(bands: list[LineBand], path: Path) -> None:
    if not bands:
        return
    thumb_w = 900
    pad = 18
    label_h = 26
    rows = []
    font = ImageFont.load_default()
    for band in bands:
        img = Image.open(band.image_path).convert("RGB")
        scale = min(thumb_w / img.width, 180 / img.height)
        thumb = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        row = Image.new("RGB", (thumb_w, label_h + thumb.height), "white")
        draw = ImageDraw.Draw(row)
        draw.text((4, 5), f"{band.line_id}  {img.width}x{img.height}", fill=(0, 0, 0), font=font)
        row.paste(thumb, (0, label_h))
        rows.append(row)

    sheet = Image.new("RGB", (thumb_w + 2 * pad, sum(r.height + pad for r in rows) + pad), (240, 240, 240))
    y = pad
    for row in rows:
        sheet.paste(row, (pad, y))
        y += row.height + pad
    sheet.save(path, quality=90)
