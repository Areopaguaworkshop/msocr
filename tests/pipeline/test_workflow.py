"""Tests for the higher-level training promotion workflow."""

import json
from pathlib import Path

from msocr.pipeline.workflow import resolve_cer_threshold, run_training_promotion_workflow


def _write_train_manifest(tmp_path: Path) -> Path:
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    xml_path = tmp_path / "corpus" / "ms001" / "line_0001.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<PcGts/>", encoding="utf-8")
    manifest_path = manifests_dir / "train.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "train-v1",
                "language": "syriac",
                "writing_mode": "printed",
                "partitions": {
                    "train": [
                        {
                            "id": "line_0001",
                            "xml_path": str(xml_path),
                            "manuscript_id": "ms001",
                        }
                    ],
                    "holdout": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_benchmark_manifest(tmp_path: Path) -> Path:
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    case_dir = tmp_path / "bench" / "ms002"
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / "page.png"
    reference_path = case_dir / "page.txt"
    image_path.write_bytes(b"png")
    reference_path.write_text("abc", encoding="utf-8")
    manifest_path = manifests_dir / "benchmark.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "bench-v1",
                "language": "syriac",
                "writing_mode": "printed",
                "partitions": {
                    "holdout": [
                        {
                            "id": "case-1",
                            "image": str(image_path),
                            "reference_text": str(reference_path),
                            "language": "syriac",
                            "manuscript_id": "ms002",
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_resolve_cer_threshold_tracks_docs_policy():
    assert resolve_cer_threshold("syriac", "estrangela", "printed") == 0.05
    assert resolve_cer_threshold("syriac", "serto", "printed") == 0.10
    assert resolve_cer_threshold("old_turkish", "old_uyghur", "printed") == 0.10
    assert resolve_cer_threshold("latin", "historical", "printed") == 0.05


def test_workflow_plan_blocks_promotion_without_benchmark(tmp_path: Path):
    train_manifest = _write_train_manifest(tmp_path)
    model_file = tmp_path / "model.traineddata"
    model_file.write_text("model", encoding="utf-8")

    plan = run_training_promotion_workflow(
        train_manifest_ref=train_manifest,
        benchmark_manifest_ref=None,
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        mode="ocr",
        trainer="tesstrain",
        config_path=None,
        pipeline_run_id="run-1",
        sequence_id="14",
        model_version="v1",
        model_file=model_file,
        registry="msocr-models",
        training_image="runpod/pytorch:demo",
        gpu_tier="auto",
        volume_in_gb=20,
        container_disk_in_gb=50,
        interruptible=False,
        network_volume_id=None,
        container_registry_auth_id=None,
        data_center_ids=(),
        dockerfile_path=None,
        output_dir=tmp_path / "output",
        cer_threshold=None,
        benchmark_id=None,
        model_id=None,
        preprocessing_profile="default",
        pkg_url="https://pkg.harness.io",
        description=None,
        metadata={},
        command=None,
        wait_for_runpod=False,
        execute=False,
    )

    assert plan["stages"]["har_publish"]["status"] == "blocked"
    assert plan["stages"]["har_publish"]["reason"] == "benchmark_manifest_required_before_promotion"


def test_workflow_execute_runs_benchmark_and_publishes(monkeypatch, tmp_path: Path):
    train_manifest = _write_train_manifest(tmp_path)
    benchmark_manifest = _write_benchmark_manifest(tmp_path)
    model_file = tmp_path / "model.traineddata"
    dockerfile = tmp_path / "Dockerfile"
    model_file.write_text("trained-model", encoding="utf-8")
    dockerfile.write_text("FROM ubuntu:22.04\n", encoding="utf-8")

    class FakeRunPodClient:
        def create_pod(self, job):
            return type(
                "Pod",
                (),
                {
                    "pod_id": "pod-123",
                    "name": job.name,
                    "desired_status": "RUNNING",
                    "public_ip": None,
                    "port_mappings": {},
                },
            )()

    published = {}

    def fake_from_env():
        return FakeRunPodClient()

    def fake_run_printed_benchmark(**kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {
            "benchmark_id": "bench-1",
            "manifest_id": "bench-v1",
            "pass_fail": True,
            "needs_manual_review": False,
            "cer": 0.01,
            "wer": 0.02,
        }

    def fake_publish_bundle(self, bundle):
        published["artifact_ref"] = bundle.artifact_ref
        published["files"] = [upload.package_path for upload in bundle.files]

    monkeypatch.setattr("msocr.pipeline.workflow.RunPodClient.from_env", fake_from_env)
    monkeypatch.setattr("msocr.pipeline.workflow.run_printed_benchmark", fake_run_printed_benchmark)
    monkeypatch.setattr("msocr.pipeline.workflow.HARClient.publish_bundle", fake_publish_bundle)

    result = run_training_promotion_workflow(
        train_manifest_ref=train_manifest,
        benchmark_manifest_ref=benchmark_manifest,
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        mode="ocr",
        trainer="tesstrain",
        config_path=None,
        pipeline_run_id="run-2",
        sequence_id="14",
        model_version="v1",
        model_file=model_file,
        registry="msocr-models",
        training_image="runpod/pytorch:demo",
        gpu_tier="auto",
        volume_in_gb=20,
        container_disk_in_gb=50,
        interruptible=False,
        network_volume_id=None,
        container_registry_auth_id=None,
        data_center_ids=(),
        dockerfile_path=dockerfile,
        output_dir=tmp_path / "output",
        cer_threshold=None,
        benchmark_id=None,
        model_id=None,
        preprocessing_profile="default",
        pkg_url="https://pkg.harness.io",
        description=None,
        metadata={"stage": "staging"},
        command=None,
        wait_for_runpod=False,
        execute=True,
    )

    assert result["stages"]["runpod"]["status"] == "submitted"
    assert result["stages"]["benchmark"]["status"] == "completed"
    assert result["stages"]["policy_gate"]["pass_fail"] is True
    assert result["stages"]["har_publish"]["status"] == "completed"
    assert published["artifact_ref"] == "syr-estrangela-printed:v14"
    assert "sidecars/metrics.json" in published["files"]
    assert "sidecars/Dockerfile.sha" in published["files"]
    assert Path(result["dockerfile_sha_path"]).exists()
