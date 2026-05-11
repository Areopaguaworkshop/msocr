"""Tests for the printed benchmark contract."""

import json
from pathlib import Path

from msocr.evaluation.printed_benchmark import run_printed_benchmark


def _write_case_files(base_dir: Path, manuscript_id: str, stem: str, text: str) -> tuple[Path, Path]:
    case_dir = base_dir / manuscript_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / f"{stem}.png"
    reference_path = case_dir / f"{stem}.txt"
    image_path.write_bytes(b"png")
    reference_path.write_text(text, encoding="utf-8")
    return image_path, reference_path


def test_run_printed_benchmark_records_manifest_contract(tmp_path: Path, monkeypatch):
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    corpus_dir = tmp_path / "corpus"
    _write_case_files(corpus_dir, "ms_train", "train_case", "ignored")
    holdout_image, holdout_reference = _write_case_files(
        corpus_dir,
        "ms_holdout",
        "holdout_case",
        "abc",
    )

    manifest_path = manifests_dir / "printed-v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "printed-v1",
                "language": "syriac",
                "writing_mode": "printed",
                "dvc_tracked": True,
                "base_dir": str(corpus_dir),
                "partitions": {
                    "train": [
                        {
                            "id": "train_case",
                            "image": "ms_train/train_case.png",
                            "reference_text": "ms_train/train_case.txt",
                            "language": "syriac",
                            "manuscript_id": "ms_train",
                        }
                    ],
                    "holdout": [
                        {
                            "id": "holdout_case",
                            "image": "ms_holdout/holdout_case.png",
                            "reference_text": "ms_holdout/holdout_case.txt",
                            "language": "syriac",
                            "script_variant": "estrangela",
                            "manuscript_id": "ms_holdout",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_run_printed_ocr(**kwargs):
        calls.append(kwargs["image_path"])
        return {"text": holdout_reference.read_text(encoding="utf-8"), "engine": "tesseract", "language": kwargs["lang"]}

    monkeypatch.setattr(
        "msocr.evaluation.printed_benchmark.run_printed_ocr",
        fake_run_printed_ocr,
    )

    output_path = tmp_path / "printed-report.json"
    report = run_printed_benchmark(
        manifest_id="printed-v1",
        manifests_dir=manifests_dir,
        output_path=output_path,
        cer_threshold=0.05,
        benchmark_id="bench-001",
        model_id="syr-printed",
        model_version="1.2.3",
        preprocessing_profile="sauvola-v1",
        pipeline_run_id="run-9",
    )

    assert calls == [holdout_image]
    assert report["benchmark_id"] == "bench-001"
    assert report["manifest_id"] == "printed-v1"
    assert report["split_version"] == "printed-v1"
    assert report["partition"] == "holdout"
    assert report["model_version"] == "1.2.3"
    assert report["pipeline_run_id"] == "run-9"
    assert report["dvc_tracked"] is True
    assert report["pass_fail"] is True
    assert report["needs_manual_review"] is False
    assert report["cer"] == 0.0
    assert report["wer"] == 0.0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["benchmark_id"] == "bench-001"
    assert written["manifest_id"] == "printed-v1"
    assert written["model_id"] == "syr-printed"
