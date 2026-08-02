"""Compile full PAGE annotations into safe Kraken recognition training XML."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from msocr.data.session_manager import decode_msocr_custom

_GAP_TOKEN_RE = re.compile(r"\[\s*gap\s*\]", re.IGNORECASE)


def _points(element: ET.Element | None) -> list[tuple[float, float]]:
    if element is None:
        return []
    return [
        (float(x), float(y))
        for x, y in (point.split(",") for point in element.get("points", "").split())
    ]


def _inside(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Return whether a point is strictly inside a polygon (ray casting)."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            boundary_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < boundary_x:
                inside = not inside
        j = i
    return inside


def _baseline_crosses_polygon(
    baseline: list[tuple[float, float]], polygon: list[list[float]]
) -> bool:
    if len(baseline) < 2 or len(polygon) < 3:
        return False
    # Sample at most one image pixel apart so a narrow lacuna cannot disappear
    # between sparse baseline vertices. Endpoints on the boundary do not count
    # as crossing; a line fragment may legitimately end at a damaged edge.
    for start, end in zip(baseline, baseline[1:]):
        sample_count = max(1, math.ceil(math.dist(start, end)))
        for step in range(1, sample_count):
            ratio = step / sample_count
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if _inside(point, polygon):
                return True
    return False


def compile_training_page_xml(
    source_xml: Path,
    output_xml: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Write recognition XML containing only eligible contiguous line samples.

    The full PAGE file remains the annotation source of truth. Excluded lines
    are physically absent from the generated XML because Kraken does not honor
    msocr's ``trainable`` metadata. Presentation token ``[gap]`` is rejected;
    lacunae belong in structural gap metadata, never recognition transcripts.
    """
    source_xml = Path(source_xml)
    output_xml = Path(output_xml)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    namespace = root.tag.split("}")[0][1:] if root.tag.startswith("{") else ""

    def tag(name: str) -> str:
        return f"{{{namespace}}}{name}" if namespace else name

    lines = list(root.iter(tag("TextLine")))
    gap_polygons: list[list[list[float]]] = []
    for line in lines:
        metadata = decode_msocr_custom(line.get("custom", ""))
        gap = metadata.get("gapAfter")
        polygon = gap.get("polygon") if isinstance(gap, dict) else None
        if isinstance(polygon, list) and len(polygon) >= 3:
            gap_polygons.append(polygon)

        unicode_el = line.find(f"{tag('TextEquiv')}/{tag('Unicode')}")
        transcript = (
            unicode_el.text if unicode_el is not None and unicode_el.text else ""
        )
        if _GAP_TOKEN_RE.search(transcript):
            raise ValueError(
                f"Line {line.get('id', '<unknown>')} contains [gap]; use structural gap metadata"
            )

    excluded: list[dict[str, str]] = []
    included = 0
    for parent in root.iter():
        for line in list(parent):
            if line.tag != tag("TextLine"):
                continue
            line_id = line.get("id", "<unknown>")
            metadata = decode_msocr_custom(line.get("custom", ""))
            baseline = _points(line.find(tag("Baseline")))
            unicode_el = line.find(f"{tag('TextEquiv')}/{tag('Unicode')}")
            transcript = (
                unicode_el.text.strip()
                if unicode_el is not None and unicode_el.text
                else ""
            )

            reason: str | None = None
            if metadata.get("trainable") is False:
                reason = str(metadata.get("exclusionReason") or "marked non-trainable")
            elif len(baseline) < 2:
                reason = "missing baseline"
            elif not transcript:
                reason = "empty transcript"
            elif any(
                _baseline_crosses_polygon(baseline, polygon) for polygon in gap_polygons
            ):
                reason = "baseline crosses declared destructive gap"

            if reason:
                parent.remove(line)
                excluded.append({"line_id": line_id, "reason": reason})
            else:
                included += 1

    audit: dict[str, Any] = {
        "source_xml": str(source_xml),
        "output_xml": str(output_xml),
        "included_count": included,
        "excluded_count": len(excluded),
        "declared_gap_polygon_count": len(gap_polygons),
        "excluded": excluded,
    }
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return audit
