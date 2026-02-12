"""Printed OCR benchmark runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from msocr.evaluation.metrics import cer, wer
from msocr.pipelines.printed_ocr import run_printed_ocr


@dataclass
class BenchmarkCase:
    image: Path
    language: str
    reference_text: Path
    engine: str = "auto"
    model: Optional[str] = None
    variant: str = "default"
    device: str = "cpu"
    manuscript_id: Optional[str] = None
    id: Optional[str] = None


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if manifest_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if (
        isinstance(parsed, dict)
        and "cases" in parsed
        and isinstance(parsed["cases"], list)
    ):
        return parsed["cases"]
    raise ValueError(
        "Unsupported manifest format. Use JSON list, {cases:[...]}, or JSONL."
    )


def _to_case(item: Dict[str, Any]) -> BenchmarkCase:
    try:
        return BenchmarkCase(
            image=Path(item["image"]),
            language=str(item["language"]).lower(),
            reference_text=Path(item["reference_text"]),
            engine=str(item.get("engine", "auto")).lower(),
            model=item.get("model"),
            variant=str(item.get("variant", "default")).lower(),
            device=str(item.get("device", "cpu")),
            manuscript_id=item.get("manuscript_id"),
            id=item.get("id"),
        )
    except KeyError as exc:
        raise ValueError(f"Manifest case missing required field: {exc}") from exc


def run_printed_benchmark(
    manifest_path: Path,
    output_path: Path,
    cer_threshold: float = 0.05,
) -> Dict[str, Any]:
    raw_items = _load_manifest(manifest_path)
    cases = [_to_case(item) for item in raw_items]
    rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        case_id = case.id or f"case_{idx:04d}"
        if not case.image.exists():
            rows.append(
                {
                    "id": case_id,
                    "language": case.language,
                    "status": "error",
                    "error": f"image_not_found:{case.image}",
                    "needs_manual_review": True,
                }
            )
            continue
        if not case.reference_text.exists():
            rows.append(
                {
                    "id": case_id,
                    "language": case.language,
                    "status": "error",
                    "error": f"reference_not_found:{case.reference_text}",
                    "needs_manual_review": True,
                }
            )
            continue

        reference = case.reference_text.read_text(encoding="utf-8")
        try:
            result = run_printed_ocr(
                lang=case.language,
                image_path=case.image,
                model=case.model,
                device=case.device,
                engine=case.engine,
                variant=case.variant,
                reference_text_path=str(case.reference_text),
                cer_threshold=cer_threshold,
            )
            hypothesis = result["text"]
            case_cer = cer(reference, hypothesis)
            case_wer = wer(reference, hypothesis)
            pass_cer = case_cer <= cer_threshold
            rows.append(
                {
                    "id": case_id,
                    "language": case.language,
                    "manuscript_id": case.manuscript_id,
                    "status": "ok",
                    "engine": result["engine"],
                    "cer": case_cer,
                    "wer": case_wer,
                    "pass_cer": pass_cer,
                    "needs_manual_review": not pass_cer,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case_id,
                    "language": case.language,
                    "manuscript_id": case.manuscript_id,
                    "status": "error",
                    "error": str(exc),
                    "needs_manual_review": True,
                }
            )

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    avg_cer = sum(r["cer"] for r in ok_rows) / len(ok_rows) if ok_rows else None
    avg_wer = sum(r["wer"] for r in ok_rows) / len(ok_rows) if ok_rows else None
    pass_count = sum(1 for r in ok_rows if r["pass_cer"])

    report: Dict[str, Any] = {
        "benchmark_type": "printed_ocr",
        "manifest": str(manifest_path),
        "cer_threshold": cer_threshold,
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "error_cases": len(rows) - len(ok_rows),
        "pass_cases": pass_count,
        "pass_rate": (pass_count / len(ok_rows)) if ok_rows else 0.0,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "rows": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
