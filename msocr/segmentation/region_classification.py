"""Region classification mapping for Payne-Smith pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _line_y(line: dict) -> int:
    bbox = line.get("bbox")
    if bbox and len(bbox) == 4:
        return int(bbox[1])
    baseline = line.get("baseline") or []
    if baseline:
        return int(min(pt[1] for pt in baseline))
    boundary = line.get("boundary") or []
    if boundary:
        return int(min(pt[1] for pt in boundary))
    return 0


def write_region_labels(segments_dir: Path, output_path: Path, region_map: Dict) -> int:
    labels = {}
    for seg_path in sorted(segments_dir.glob("*.segments.json")):
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        lines = data.get("lines", [])
        ordered = sorted(enumerate(lines, start=1), key=lambda pair: _line_y(pair[1]))
        title_line_indexes = {ordered[0][0]} if ordered else set()

        for idx, _line in enumerate(lines, start=1):
            region_type = "TitleZone" if idx in title_line_indexes else "MainZone"
            line_id = f"{Path(data['image']).stem}_{idx:06d}"
            labels[line_id] = {
                "region_type": region_type,
                "script": region_map.get(region_type, region_map.get("MainZone", "Serto")),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(labels)
