"""Printed OCR benchmark runner."""

from __future__ import annotations

from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from msocr.data.manifest import iter_partition_cases, load_frozen_manifest
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


def _summarize_identity(values: List[str], fallback: str) -> str:
    normalized = [value for value in values if value]
    if not normalized:
        return fallback
    unique_values = sorted(set(normalized))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def _to_case(
    item: Any,
    *,
    default_language: Optional[str] = None,
    default_engine: Optional[str] = None,
    default_model: Optional[str] = None,
    default_variant: Optional[str] = None,
    default_device: Optional[str] = None,
) -> BenchmarkCase:
    if item.image is None:
        raise ValueError(f"Benchmark case {item.id!r} is missing image path")
    if item.reference_text is None:
        raise ValueError(f"Benchmark case {item.id!r} is missing reference_text path")
    language = default_language or item.language
    if not language:
        raise ValueError(f"Benchmark case {item.id!r} is missing language")

    return BenchmarkCase(
        image=item.image,
        language=language,
        reference_text=item.reference_text,
        engine=default_engine or item.engine,
        model=default_model if default_model is not None else item.model,
        variant=default_variant or item.variant,
        device=default_device or item.device,
        manuscript_id=item.manuscript_id,
        id=item.id,
    )


def run_printed_benchmark(
    *,
    output_path: Path,
    manifest_path: Optional[Path] = None,
    manifest_id: Optional[str] = None,
    cer_threshold: float = 0.05,
    benchmark_id: Optional[str] = None,
    model_id: str = "printed_ocr",
    model_version: str = "local",
    preprocessing_profile: str = "default",
    pipeline_run_id: str = "local",
    manifests_dir: Optional[Path] = None,
    default_language: Optional[str] = None,
    default_engine: Optional[str] = None,
    default_model: Optional[str] = None,
    default_variant: Optional[str] = None,
    default_device: Optional[str] = None,
) -> Dict[str, Any]:
    if manifest_path is not None and manifest_id is not None:
        raise ValueError("Provide manifest_path or manifest_id, not both.")
    if manifest_path is None and manifest_id is None:
        raise ValueError("A manifest_path or manifest_id is required.")

    manifest_ref = manifest_path if manifest_path is not None else manifest_id
    manifest = load_frozen_manifest(manifest_ref, manifests_dir=manifests_dir)

    selected_cases = iter_partition_cases(
        manifest,
        ("holdout", "cases", "validation", "train"),
    )
    if manifest.partitions.get("holdout"):
        partition_name = "holdout"
    elif manifest.partitions.get("cases"):
        partition_name = "cases"
    elif manifest.partitions.get("validation"):
        partition_name = "validation"
    else:
        partition_name = "train"

    cases = [
        _to_case(
            item,
            default_language=default_language,
            default_engine=default_engine,
            default_model=default_model,
            default_variant=default_variant,
            default_device=default_device,
        )
        for item in selected_cases
    ]
    rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        case_id = case.id or f"case_{idx:04d}"
        if not case.image.exists():
            rows.append(
                {
                    "id": case_id,
                    "language": case.language,
                    "script_variant": case.variant,
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
                    "script_variant": case.variant,
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
                    "script_variant": case.variant,
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
                    "script_variant": case.variant,
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
    pass_fail = bool(ok_rows) and len(ok_rows) == len(rows) and pass_count == len(ok_rows)
    case_languages = [case.language for case in cases]
    case_variants = [case.variant for case in cases]
    report_language = manifest.language or _summarize_identity(case_languages, "unknown")
    report_variant = _summarize_identity(case_variants, "default")
    resolved_benchmark_id = benchmark_id or manifest.manifest_id

    report: Dict[str, Any] = {
        "benchmark_type": "printed_ocr",
        "benchmark_id": resolved_benchmark_id,
        "manifest": str(manifest.path),
        "manifest_id": manifest.manifest_id,
        "split_version": manifest.manifest_id,
        "partition": partition_name,
        "writing_mode": manifest.writing_mode,
        "language": report_language,
        "script_variant": report_variant,
        "model_id": model_id,
        "model_version": model_version,
        "preprocessing_profile": preprocessing_profile,
        "pipeline_run_id": pipeline_run_id,
        "dvc_tracked": manifest.dvc_tracked,
        "generated_at": datetime.now().isoformat(),
        "cer_threshold": cer_threshold,
        "cer": avg_cer,
        "wer": avg_wer,
        "pass_fail": pass_fail,
        "needs_manual_review": not pass_fail,
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
