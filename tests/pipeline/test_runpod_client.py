"""Tests for RunPod orchestration helpers."""

import json
from pathlib import Path

from msocr.data.manifest import load_frozen_manifest
from msocr.pipeline.runpod_client import build_training_job, recommend_gpu_tier


def _write_manifest(tmp_path: Path, *, entries: int) -> Path:
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    corpus_dir = tmp_path / "corpus"
    train_entries = []
    for index in range(entries):
        manuscript_id = f"ms{index:04d}"
        case_dir = corpus_dir / manuscript_id
        case_dir.mkdir(parents=True, exist_ok=True)
        xml_path = case_dir / f"line_{index:04d}.xml"
        xml_path.write_text("<PcGts/>", encoding="utf-8")
        train_entries.append(
            {
                "id": xml_path.stem,
                "xml_path": str(xml_path),
                "manuscript_id": manuscript_id,
            }
        )

    manifest_path = manifests_dir / "training-v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "training-v1",
                "language": "syriac",
                "writing_mode": "printed",
                "partitions": {"train": train_entries, "holdout": []},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_recommend_gpu_tier_tracks_backend_and_corpus_size():
    assert recommend_gpu_tier(corpus_size=500, training_backend="tesstrain") == "rtx4090"
    assert recommend_gpu_tier(corpus_size=2500, training_backend="kraken") == "rtx4090"
    assert recommend_gpu_tier(corpus_size=25001, training_backend="kraken") == "a100"


def test_build_training_job_uses_manifest_contract(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, entries=3)
    manifest = load_frozen_manifest(manifest_path, manifests_dir=manifest_path.parent)

    job = build_training_job(
        manifest=manifest,
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        mode="ocr",
        pipeline_run_id="run-42",
        model_version="v3",
        partition="train",
    )

    payload = job.to_api_payload()
    assert payload["name"].startswith("msocr-syr-estrangela-printed-run-42")
    assert payload["gpuTypeIds"][0] == "NVIDIA GeForce RTX 4090"
    assert payload["env"]["MSOCR_MANIFEST_ID"] == "training-v1"
    assert payload["env"]["MSOCR_CORPUS_SIZE"] == "3"
    assert payload["dockerStartCmd"][-1].endswith(
        "--split-manifest-id training-v1 --split-partition train"
    )
