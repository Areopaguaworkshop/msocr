"""Bootstrap OCR for line images using Kraken model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from msocr.models.inference import OCRModel


def _aggregate(confidences: list[float], method: str) -> float:
    if not confidences:
        return 0.0
    if method == "min":
        return min(confidences)
    if method == "p10":
        confidences = sorted(confidences)
        idx = max(0, int(len(confidences) * 0.10) - 1)
        return confidences[idx]
    return sum(confidences) / len(confidences)


def bootstrap_ocr_lines(
    lines_dir: Path,
    model_path: Path,
    bootstrap_dir: Path,
    rejected_dir: Path,
    cfg: Dict,
) -> int:
    model = OCRModel(model_path)
    count = 0
    method = cfg.get("confidence_aggregation", "mean")
    threshold = float(cfg.get("confidence_threshold", 0.8))
    segmentation_type = cfg.get("segmentation_type", "bbox")

    for image_path in sorted(lines_dir.glob("*.png")):
        # Line crops are already segmented; bbox avoids expensive re-segmentation.
        result = model.predict_line(image_path, segmentation_type=segmentation_type)
        preds = result.get("predictions", [])
        confs = [p.get("confidence", 0.0) for p in preds]
        agg = _aggregate(confs, method)
        item = {
            "line_id": image_path.stem,
            "image_path": str(image_path),
            "transcription": result.get("full_text", ""),
            "bootstrap_confidence": agg,
        }
        target_dir = bootstrap_dir if agg >= threshold else rejected_dir
        out_json = target_dir / f"{image_path.stem}.json"
        out_json.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count
