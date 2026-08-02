"""CC-based fragment isolation for Sogdian manuscript pages (Phase 1).

ponytail: Sauvola threshold + connectedComponents + DBSCAN on component
centroids. Replaces the union-bbox approach in manuscript_area.py, which
remains as a fallback. One fragment may hold multiple rows; lines come later
via Kraken (Phase 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_sauvola
from sklearn.cluster import DBSCAN

from msocr.segmentation.row_bands import _clamp_roi


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)
    area: int
    flagged: str | None


def _contiguous_groups(values: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        number = int(value)
        if not groups or number > groups[-1][-1] + 1:
            groups.append([number])
        else:
            groups[-1].append(number)
    return groups


def _mounted_inner_frame(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Find the inside of the dark DTA photographic frame."""
    height, width = gray.shape
    dark = gray < 50
    row_density = dark.mean(axis=1)
    column_density = dark.mean(axis=0)

    def _edge(density: np.ndarray, length: int, start: bool) -> int:
        indexes = np.where(density > 0.65)[0]
        indexes = (
            indexes[indexes < length * 0.18]
            if start
            else indexes[indexes > length * 0.82]
        )
        groups = [group for group in _contiguous_groups(indexes) if len(group) >= 3]
        if not groups:
            return 0 if start else length
        band = max(
            groups,
            key=lambda group: len(group) * float(density[group].mean()),
        )
        return band[-1] + 1 if start else band[0]

    return (
        _edge(column_density, width, True),
        _edge(row_density, height, True),
        _edge(column_density, width, False),
        _edge(row_density, height, False),
    )


def isolate_mounted_fragment(
    image: Image.Image,
) -> tuple[Fragment, np.ndarray] | None:
    """Isolate the central physical fragment in a mounted DTA photograph.

    The E27 downloads contain one target fragment on a pale mount plus a dark
    frame, labels, and often a ruler. Ink-component DBSCAN cannot separate
    those reliably. This DTA-specific path removes the frame, seeds GrabCut
    from parchment/ink color, rejects components connected to the frame, and
    chooses the large component nearest the image center. The returned mask is
    full-page sized and must still pass human overlay QA before annotation.
    """
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = _mounted_inner_frame(gray)
    if right - left < 10 or bottom - top < 10:
        return None

    crop = bgr[top:bottom, left:right]
    height, width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    grabcut_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    pad = max(3, int(min(width, height) * 0.025))
    grabcut_mask[:pad] = cv2.GC_BGD
    grabcut_mask[-pad:] = cv2.GC_BGD
    grabcut_mask[:, :pad] = cv2.GC_BGD
    grabcut_mask[:, -pad:] = cv2.GC_BGD
    probable_foreground = (value < 210) | (saturation > 65)
    grabcut_mask[probable_foreground] = cv2.GC_PR_FGD
    grabcut_mask[(value < 100) & probable_foreground] = cv2.GC_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        crop,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel_size = max(7, int(min(width, height) * 0.015) // 2 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if area < height * width * 0.002:
            continue
        if (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        ):
            continue
        center_x, center_y = centroids[label]
        distance = ((center_x - width / 2) / (width / 2)) ** 2 + (
            (center_y - height / 2) / (height / 2)
        ) ** 2
        candidates.append((area / (1 + distance * 2), label))
    if not candidates:
        return None

    _, selected_label = max(candidates)
    x, y, component_width, component_height, area = (
        int(value) for value in stats[selected_label]
    )
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    full_mask[top:bottom, left:right] = np.where(
        labels == selected_label, 255, 0
    ).astype(np.uint8)
    bbox = _clamp_roi(
        (
            left + x - 15,
            top + y - 15,
            left + x + component_width + 15,
            top + y + component_height + 15,
        ),
        image.size,
    )
    return (
        Fragment(
            fragment_id="frag_001",
            bbox=bbox,
            area=area,
            flagged=None,
        ),
        full_mask,
    )


def _sauvola_ink_components(
    image: Image.Image, *, window: int, min_area: int
) -> list[tuple[int, int, int, int, int]]:
    """Sauvola-threshold foreground components as ``(x, y, x+w, y+h, area)``."""
    gray = np.array(image.convert("L"))
    # skimage expects float in [0, 1] or uint8; uint8 works with window_size in px.
    t = threshold_sauvola(gray, window_size=window)
    # ink is darker than the local threshold -> foreground mask
    binary = (gray < t).astype(np.uint8) * 255
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= min_area:
            out.append((x, y, x + w, y + h, area))
    return out


def isolate_fragments(
    image: Image.Image,
    *,
    sauvola_window: int = 51,
    min_component_area: int = 50,
    min_fragment_area: int = 5000,
    dbscan_eps: int = 150,
    dbscan_min_samples: int = 1,
) -> list[Fragment]:
    """Find manuscript fragments via Sauvola + CC + DBSCAN.

    Returns fragments in top-to-bottom, left-to-right scan order.
    """
    components = _sauvola_ink_components(
        image, window=sauvola_window, min_area=min_component_area
    )
    if not components:
        return []

    centroids = np.array(
        [[((c[0] + c[2]) / 2), ((c[1] + c[3]) / 2)] for c in components],
        dtype=np.float32,
    )
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(centroids)
    labels = db.labels_

    raw: list[tuple[int, int, int, int, int, str]] = []  # (top, left, r, b, area, fid)
    for label in set(labels):
        if label == -1:  # DBSCAN noise with min_samples=1 shouldn't happen, but be safe
            continue
        idxs = [i for i, lb in enumerate(labels) if lb == label]
        cluster = [components[i] for i in idxs]
        left = min(c[0] for c in cluster)
        top = min(c[1] for c in cluster)
        right = max(c[2] for c in cluster)
        bottom = max(c[3] for c in cluster)
        area = sum(c[4] for c in cluster)
        raw.append((top, left, right, bottom, area, ""))

    # scan order: top, then left
    raw.sort(key=lambda r: (r[0], r[1]))

    fragments: list[Fragment] = []
    for order, (top, left, right, bottom, area, _) in enumerate(raw, start=1):
        bbox = _clamp_roi((left - 20, top - 20, right + 20, bottom + 20), image.size)
        flagged = "FRAGMENT_TOO_SMALL" if area < min_fragment_area else None
        fragments.append(
            Fragment(
                fragment_id=f"frag_{order:03d}",
                bbox=bbox,
                area=area,
                flagged=flagged,
            )
        )
    return fragments


def write_fragment_overlay(
    image: Image.Image, fragments: list[Fragment], output_path: Path
) -> None:
    """Draw fragment bboxes (red normal, orange too-small) with id labels."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for frag in fragments:
        color = (255, 165, 0) if frag.flagged == "FRAGMENT_TOO_SMALL" else (220, 30, 30)
        draw.rectangle(frag.bbox, outline=color, width=3)
        draw.text(
            (frag.bbox[0] + 4, frag.bbox[1] + 4),
            frag.fragment_id,
            fill=color,
            font=font,
        )
    overlay.save(output_path, format="JPEG", quality=90)


def fragments_to_json(fragments: list[Fragment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "fragment_id": f.fragment_id,
            "bbox": list(f.bbox),
            "area": f.area,
            "flagged": f.flagged,
        }
        for f in fragments
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fragments_from_json(path: Path) -> list[Fragment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Fragment(
            fragment_id=item["fragment_id"],
            bbox=tuple(item["bbox"]),  # type: ignore[arg-type]
            area=int(item["area"]),
            flagged=item["flagged"],
        )
        for item in payload
    ]


if __name__ == "__main__":
    # Self-check: three dark rectangles + one tiny fleck on white.
    page = Image.new("L", (800, 600), 255)
    rects = [(80, 80, 220, 160), (380, 260, 540, 360), (120, 440, 280, 520)]
    for x0, y0, x1, y1 in rects:
        for y in range(y0, y1):
            for x in range(x0, x1):
                page.putpixel((x, y), 20)
    # tiny fleck — below min_component_area=50 default
    for y in range(700, 712):
        for x in range(490, 502):
            try:
                page.putpixel((x, y), 20)
            except IndexError:
                pass

    fragments = isolate_fragments(
        page, sauvola_window=51, min_component_area=50, min_fragment_area=5000
    )
    assert (
        len(fragments) == 3
    ), f"expected 3 fragments, got {len(fragments)}: {fragments}"

    for frag, (x0, y0, x1, y1) in zip(fragments, rects):
        left, top, right, bottom = frag.bbox
        assert left <= x0 and top <= y0 and right >= x1 and bottom >= y1, (
            frag,
            (x0, y0, x1, y1),
        )
        assert frag.area >= 5000, frag
        assert frag.flagged is None, frag

    # no fragment is the whole image
    for frag in fragments:
        left, top, right, bottom = frag.bbox
        assert (right - left) < 800 or (bottom - top) < 600, frag

    print("ok")
