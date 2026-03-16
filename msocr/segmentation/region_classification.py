"""Region classification mapping for Payne-Smith pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_region_labels(segments_dir: Path, output_path: Path, region_map: Dict) -> int:
    labels = {}
    for seg_path in sorted(segments_dir.glob("*.segments.json")):
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        for idx, _line in enumerate(data.get("lines", []), start=1):
            line_id = f"{Path(data['image']).stem}_{idx:06d}"
            labels[line_id] = {
                "region_type": "MainZone",
                "script": region_map.get("MainZone", "Serto"),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(labels)
