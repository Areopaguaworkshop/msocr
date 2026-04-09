"""Higher-level manifest-aware training promotion workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from msocr.data.manifest import FrozenManifest, load_frozen_manifest
from msocr.evaluation.printed_benchmark import run_printed_benchmark
from msocr.language_registry import normalize_language_code
from msocr.pipeline.har_client import HARClient, build_bundle, build_model_artifact_name
from msocr.pipeline.runpod_client import RunPodClient, build_training_job


DEFAULT_PIPELINE_OUTPUT_DIR = Path("output/pipelines")


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def resolve_cer_threshold(language: str, script_variant: str, writing_mode: str) -> float:
    normalized_language = normalize_language_code(language)
    normalized_variant = _slug(script_variant)
    normalized_writing_mode = _slug(writing_mode)

    if normalized_writing_mode == "handwritten":
        return 0.10
    if normalized_language == "syriac":
        if normalized_variant in {"serto", "east", "east-syriac", "east-syriac"}:
            return 0.10
        return 0.05
    if normalized_language in {"sogdian", "old_turkish"}:
        return 0.10
    return 0.05


def _infer_engine_override(trainer: str, model_file: Optional[Path]) -> Optional[str]:
    trainer_key = _slug(trainer)
    if trainer_key in {"tesstrain", "tesseract"}:
        return "tesseract"
    if trainer_key == "kraken":
        return "kraken"
    if model_file is None:
        return None
    if model_file.suffix.lower() == ".traineddata":
        return "tesseract"
    if model_file.suffix.lower() == ".mlmodel":
        return "kraken"
    return None


def write_dockerfile_sha(dockerfile_path: Path, output_path: Path) -> Path:
    digest = hashlib.sha256(dockerfile_path.read_bytes()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(digest, encoding="utf-8")
    return output_path


def _artifact_version(sequence_id: Optional[str], pipeline_run_id: str) -> str:
    token = sequence_id.strip() if sequence_id else _slug(pipeline_run_id)
    return f"v{token}"


def _build_output_paths(
    *,
    output_dir: Path,
    pipeline_run_id: str,
    artifact_name: str,
) -> Dict[str, Path]:
    root = output_dir / pipeline_run_id / artifact_name
    return {
        "root": root,
        "metrics": root / "metrics.json",
        "dockerfile_sha": root / "Dockerfile.sha",
    }


def _load_manifest(reference: str | Path) -> FrozenManifest:
    return load_frozen_manifest(reference)


def run_training_promotion_workflow(
    *,
    train_manifest_ref: str | Path,
    benchmark_manifest_ref: Optional[str | Path],
    language: str,
    script_variant: str,
    writing_mode: str,
    mode: str,
    trainer: str,
    config_path: Optional[Path],
    pipeline_run_id: str,
    sequence_id: Optional[str],
    model_version: str,
    model_file: Optional[Path],
    registry: Optional[str],
    training_image: str,
    gpu_tier: str,
    volume_in_gb: int,
    container_disk_in_gb: int,
    interruptible: bool,
    network_volume_id: Optional[str],
    container_registry_auth_id: Optional[str],
    data_center_ids: Iterable[str],
    dockerfile_path: Optional[Path],
    output_dir: Path,
    cer_threshold: Optional[float],
    benchmark_id: Optional[str],
    model_id: Optional[str],
    preprocessing_profile: str,
    pkg_url: str,
    description: Optional[str],
    metadata: Dict[str, str],
    command: Optional[str],
    wait_for_runpod: bool,
    execute: bool,
) -> Dict[str, Any]:
    train_manifest = _load_manifest(train_manifest_ref)
    benchmark_manifest = _load_manifest(benchmark_manifest_ref) if benchmark_manifest_ref else None
    normalized_language = normalize_language_code(language)
    normalized_writing_mode = _slug(writing_mode)
    normalized_mode = mode.strip().lower()
    artifact_name = build_model_artifact_name(
        normalized_language,
        script_variant,
        normalized_writing_mode,
    )
    artifact_version = _artifact_version(sequence_id, pipeline_run_id)
    output_paths = _build_output_paths(
        output_dir=output_dir,
        pipeline_run_id=pipeline_run_id,
        artifact_name=artifact_name,
    )
    resolved_threshold = cer_threshold or resolve_cer_threshold(
        normalized_language,
        script_variant,
        normalized_writing_mode,
    )

    training_job = build_training_job(
        manifest=train_manifest,
        language=normalized_language,
        script_variant=script_variant,
        writing_mode=normalized_writing_mode,
        mode=normalized_mode,
        pipeline_run_id=pipeline_run_id,
        model_version=model_version,
        training_image=training_image,
        training_backend=None if _slug(trainer) == "auto" else trainer,
        config_path=config_path,
        partition="train",
        gpu_tier=gpu_tier,
        command=command,
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        interruptible=interruptible,
        network_volume_id=network_volume_id,
        container_registry_auth_id=container_registry_auth_id,
        data_center_ids=data_center_ids,
    )

    result: Dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "language": normalized_language,
        "script_variant": script_variant,
        "writing_mode": normalized_writing_mode,
        "mode": normalized_mode,
        "trainer": _slug(trainer),
        "train_manifest_id": train_manifest.manifest_id,
        "benchmark_manifest_id": benchmark_manifest.manifest_id if benchmark_manifest else None,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "output_root": str(output_paths["root"]),
        "stages": {
            "runpod": {
                "status": "planned",
                "payload": training_job.to_api_payload(),
            },
            "benchmark": {
                "status": "skipped" if benchmark_manifest is None else "planned",
                "output_path": str(output_paths["metrics"]),
                "cer_threshold": resolved_threshold,
            },
            "policy_gate": {
                "status": "skipped" if benchmark_manifest is None else "planned",
                "cer_threshold": resolved_threshold,
            },
            "har_publish": {
                "status": "skipped" if not registry or model_file is None else "planned",
                "registry": registry,
                "pkg_url": pkg_url,
            },
        },
    }

    if dockerfile_path and dockerfile_path.exists():
        result["dockerfile"] = str(dockerfile_path)
        result["dockerfile_sha_path"] = str(output_paths["dockerfile_sha"])

    if not execute:
        if registry and model_file is not None and benchmark_manifest is None:
            result["stages"]["har_publish"] = {
                **result["stages"]["har_publish"],
                "status": "blocked",
                "reason": "benchmark_manifest_required_before_promotion",
            }
        return result

    client = RunPodClient.from_env()
    pod = client.create_pod(training_job)
    result["stages"]["runpod"] = {
        **result["stages"]["runpod"],
        "status": "submitted",
        "pod": {
            "id": pod.pod_id,
            "name": pod.name,
            "desired_status": pod.desired_status,
            "public_ip": pod.public_ip,
            "port_mappings": pod.port_mappings,
        },
    }
    if wait_for_runpod:
        ready_pod = client.wait_for_status(pod.pod_id)
        result["stages"]["runpod"] = {
            **result["stages"]["runpod"],
            "status": "running",
            "pod": {
                "id": ready_pod.pod_id,
                "name": ready_pod.name,
                "desired_status": ready_pod.desired_status,
                "public_ip": ready_pod.public_ip,
                "port_mappings": ready_pod.port_mappings,
            },
        }

    benchmark_report: Optional[Dict[str, Any]] = None
    if benchmark_manifest is not None:
        benchmark_report = run_printed_benchmark(
            output_path=output_paths["metrics"],
            manifest_path=benchmark_manifest.path,
            cer_threshold=resolved_threshold,
            benchmark_id=benchmark_id,
            model_id=model_id or artifact_name,
            model_version=model_version,
            preprocessing_profile=preprocessing_profile,
            pipeline_run_id=pipeline_run_id,
            default_language=normalized_language,
            default_engine=_infer_engine_override(trainer, model_file),
            default_model=str(model_file) if model_file is not None else None,
            default_variant=script_variant,
        )
        result["stages"]["benchmark"] = {
            **result["stages"]["benchmark"],
            "status": "completed",
            "report": benchmark_report,
        }
        result["stages"]["policy_gate"] = {
            "status": "completed",
            "cer_threshold": resolved_threshold,
            "pass_fail": benchmark_report["pass_fail"],
            "needs_manual_review": benchmark_report["needs_manual_review"],
        }

    if not registry or model_file is None:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "skipped",
            "reason": "registry_and_model_file_required",
        }
        return result

    if benchmark_report is None:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "blocked",
            "reason": "benchmark_manifest_required_before_promotion",
        }
        return result

    if not benchmark_report["pass_fail"]:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "blocked",
            "reason": "policy_gate_failed",
        }
        return result

    dockerfile_sha_file = None
    if dockerfile_path and dockerfile_path.exists():
        dockerfile_sha_file = write_dockerfile_sha(
            dockerfile_path,
            output_paths["dockerfile_sha"],
        )

    bundle = build_bundle(
        registry=registry,
        language=normalized_language,
        script_variant=script_variant,
        writing_mode=normalized_writing_mode,
        version=artifact_version,
        model_file=model_file,
        metrics_file=output_paths["metrics"] if output_paths["metrics"].exists() else None,
        config_file=config_path if config_path and config_path.exists() else None,
        dockerfile_sha_file=dockerfile_sha_file,
        pkg_url=pkg_url,
        description=description,
        metadata=metadata,
    )
    har_client = HARClient(pkg_url=pkg_url)
    planned_commands = [" ".join(command) for command in har_client.plan_commands(bundle)]
    har_client.publish_bundle(bundle)
    result["stages"]["har_publish"] = {
        **result["stages"]["har_publish"],
        "status": "completed",
        "artifact_ref": bundle.artifact_ref,
        "commands": planned_commands,
    }
    return result
