"""Higher-level manifest-aware training promotion workflow."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional
from urllib import error, request

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
        "notification": root / "notification.json",
        "state": root / "state.json",
        "retrieved": root / "retrieved",
    }


def _load_manifest(reference: str | Path) -> FrozenManifest:
    return load_frozen_manifest(reference)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _delegate_runtime_context() -> Dict[str, str]:
    env_map = {
        "HARNESS_ACCOUNT_ID": "account_id",
        "HARNESS_ORG_ID": "org_id",
        "HARNESS_PROJECT_ID": "project_id",
        "HARNESS_BUILD_ID": "build_id",
        "HARNESS_EXECUTION_ID": "execution_id",
        "CI_BUILD_LINK": "build_url",
    }
    context: Dict[str, str] = {}
    for env_name, key in env_map.items():
        value = os.getenv(env_name)
        if value:
            context[key] = value
    return context


def _planned_retrieval_path(output_root: Path, remote_path: str) -> Path:
    filename = PurePosixPath(remote_path).name
    if not filename:
        raise ValueError(f"RunPod model path must point to a file: {remote_path}")
    return output_root / "retrieved" / filename


def _build_notification(
    result: Dict[str, Any],
    *,
    event: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "event": event,
        "message": message,
        "created_at": _utc_timestamp(),
        "pipeline_run_id": result["pipeline_run_id"],
        "artifact_name": result["artifact_name"],
        "artifact_version": result["artifact_version"],
        "language": result["language"],
        "script_variant": result["script_variant"],
        "writing_mode": result["writing_mode"],
        "train_manifest_id": result["train_manifest_id"],
        "benchmark_manifest_id": result["benchmark_manifest_id"],
        "delegate_context": result.get("delegate_context", {}),
        "stages": result["stages"],
    }


def _deliver_notification(notification_url: str, payload: Dict[str, Any]) -> None:
    req = request.Request(
        notification_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30):
            return
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Notification delivery failed: {exc.code} {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Notification delivery failed: {exc.reason}") from exc


def _emit_notification(
    result: Dict[str, Any],
    *,
    output_paths: Dict[str, Path],
    event: str,
    message: str,
    notification_url: Optional[str],
) -> None:
    payload = _build_notification(result, event=event, message=message)
    _write_json(output_paths["notification"], payload)
    result["notification_event"] = event
    result["notification_path"] = str(output_paths["notification"])
    if not notification_url:
        result["notification_delivery"] = {"status": "written"}
        return
    try:
        _deliver_notification(notification_url, payload)
        result["notification_delivery"] = {
            "status": "delivered",
            "url": notification_url,
        }
    except RuntimeError as exc:
        result["notification_delivery"] = {
            "status": "failed",
            "url": notification_url,
            "error": str(exc),
        }


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
    runpod_model_path: Optional[str],
    runpod_ssh_key_path: Optional[Path],
    runpod_ssh_public_key: Optional[str],
    runpod_retrieve_timeout_sec: int,
    runpod_retrieve_poll_interval_sec: int,
    notification_url: Optional[str],
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
    planned_retrieved_model = (
        _planned_retrieval_path(output_paths["root"], runpod_model_path)
        if runpod_model_path
        else None
    )
    resolved_threshold = cer_threshold or resolve_cer_threshold(
        normalized_language,
        script_variant,
        normalized_writing_mode,
    )
    has_model_source = model_file is not None or runpod_model_path is not None

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
        enable_ssh=model_file is None and runpod_model_path is not None,
        ssh_public_key=runpod_ssh_public_key,
    )

    result: Dict[str, Any] = {
        "created_at": _utc_timestamp(),
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
        "state_path": str(output_paths["state"]),
        "notification_path": str(output_paths["notification"]),
        "delegate_context": _delegate_runtime_context(),
        "model_file": str(model_file) if model_file is not None else None,
        "stages": {
            "runpod": {
                "status": "planned",
                "payload": training_job.to_api_payload(),
            },
            "model_retrieval": {
                "status": "skipped"
                if model_file is not None or runpod_model_path is None
                else "planned",
                "remote_path": runpod_model_path,
                "local_path": str(planned_retrieved_model) if planned_retrieved_model else None,
            },
            "benchmark": {
                "status": "skipped"
                if benchmark_manifest is None
                else ("planned" if has_model_source else "blocked"),
                "output_path": str(output_paths["metrics"]),
                "cer_threshold": resolved_threshold,
                **(
                    {"reason": "model_artifact_required_for_benchmark"}
                    if benchmark_manifest is not None and not has_model_source
                    else {}
                ),
            },
            "policy_gate": {
                "status": "skipped"
                if benchmark_manifest is None
                else ("planned" if has_model_source else "blocked"),
                "cer_threshold": resolved_threshold,
                **(
                    {"reason": "model_artifact_required_for_benchmark"}
                    if benchmark_manifest is not None and not has_model_source
                    else {}
                ),
            },
            "har_publish": {
                "status": "skipped"
                if not registry
                else (
                    "planned"
                    if benchmark_manifest is not None and has_model_source
                    else "blocked"
                ),
                "registry": registry,
                "pkg_url": pkg_url,
                **(
                    {"reason": "benchmark_manifest_required_before_promotion"}
                    if registry and benchmark_manifest is None
                    else {}
                ),
                **(
                    {"reason": "model_artifact_required_before_promotion"}
                    if registry and benchmark_manifest is not None and not has_model_source
                    else {}
                ),
            },
        },
    }

    if dockerfile_path and dockerfile_path.exists():
        result["dockerfile"] = str(dockerfile_path)
        result["dockerfile_sha_path"] = str(output_paths["dockerfile_sha"])

    def persist_state() -> None:
        _write_json(output_paths["state"], result)

    if not execute:
        if registry and model_file is not None and benchmark_manifest is None:
            result["stages"]["har_publish"] = {
                **result["stages"]["har_publish"],
                "status": "blocked",
                "reason": "benchmark_manifest_required_before_promotion",
            }
        persist_state()
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
    persist_state()

    active_pod = pod
    if wait_for_runpod or (model_file is None and runpod_model_path is not None):
        ready_pod = client.wait_for_status(pod.pod_id)
        active_pod = ready_pod
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
        persist_state()

    resolved_model_file = model_file
    if resolved_model_file is None and runpod_model_path is not None and planned_retrieved_model is not None:
        try:
            resolved_model_file = client.retrieve_model(
                pod=active_pod,
                remote_path=runpod_model_path,
                local_path=planned_retrieved_model,
                ssh_key_path=runpod_ssh_key_path,
                timeout_sec=runpod_retrieve_timeout_sec,
                poll_interval_sec=runpod_retrieve_poll_interval_sec,
            )
        except (RuntimeError, TimeoutError) as exc:
            result["stages"]["model_retrieval"] = {
                **result["stages"]["model_retrieval"],
                "status": "failed",
                "error": str(exc),
            }
            persist_state()
            _emit_notification(
                result,
                output_paths=output_paths,
                event="model_retrieval_failed",
                message="RunPod model retrieval failed before benchmark evaluation.",
                notification_url=notification_url,
            )
            persist_state()
            raise
        result["model_file"] = str(resolved_model_file)
        result["stages"]["model_retrieval"] = {
            **result["stages"]["model_retrieval"],
            "status": "completed",
            "local_path": str(resolved_model_file),
        }
        persist_state()

    benchmark_report: Optional[Dict[str, Any]] = None
    if benchmark_manifest is not None:
        if resolved_model_file is None:
            result["stages"]["benchmark"] = {
                **result["stages"]["benchmark"],
                "status": "blocked",
                "reason": "model_artifact_required_for_benchmark",
            }
            result["stages"]["policy_gate"] = {
                **result["stages"]["policy_gate"],
                "status": "blocked",
                "reason": "model_artifact_required_for_benchmark",
            }
            if registry:
                result["stages"]["har_publish"] = {
                    **result["stages"]["har_publish"],
                    "status": "blocked",
                    "reason": "model_artifact_required_before_promotion",
                }
            persist_state()
            _emit_notification(
                result,
                output_paths=output_paths,
                event="workflow_blocked",
                message="Benchmark and promotion were blocked because no trained model artifact was available locally or from RunPod.",
                notification_url=notification_url,
            )
            persist_state()
            return result

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
            default_engine=_infer_engine_override(trainer, resolved_model_file),
            default_model=str(resolved_model_file),
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
        persist_state()

    if not registry or resolved_model_file is None:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "skipped",
            "reason": "registry_and_model_file_required",
        }
        persist_state()
        return result

    if benchmark_report is None:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "blocked",
            "reason": "benchmark_manifest_required_before_promotion",
        }
        persist_state()
        return result

    if not benchmark_report["pass_fail"]:
        result["stages"]["har_publish"] = {
            **result["stages"]["har_publish"],
            "status": "blocked",
            "reason": "policy_gate_failed",
        }
        persist_state()
        _emit_notification(
            result,
            output_paths=output_paths,
            event="policy_gate_failed",
            message="Benchmark CER policy gate failed; artifact promotion was blocked.",
            notification_url=notification_url,
        )
        persist_state()
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
        model_file=resolved_model_file,
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
    persist_state()
    _emit_notification(
        result,
        output_paths=output_paths,
        event="artifact_promoted",
        message="Artifact promotion completed successfully.",
        notification_url=notification_url,
    )
    persist_state()
    return result
