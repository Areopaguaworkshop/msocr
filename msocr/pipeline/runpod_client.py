"""RunPod orchestration client for manifest-driven training jobs."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib import error, request

from msocr.data.manifest import FrozenManifest
from msocr.language_registry import LANGUAGE_REGISTRY, normalize_language_code


RUNPOD_BASE_URL = "https://rest.runpod.io/v1"
DEFAULT_TRAINING_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

GPU_TIER_CHOICES = {
    "rtx4090": ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"),
    "rtx3090": ("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"),
    # RunPod's current public REST documentation lists 80 GB A100 variants.
    "a100": ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"),
}

_SSH_PORT_KEYS = ("22/tcp", "22", "tcp/22")


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _language_token(language: str) -> str:
    metadata = LANGUAGE_REGISTRY.get(language, {})
    return str(metadata.get("iso", language)).lower()


def _artifact_token(language: str, script_variant: str, writing_mode: str) -> str:
    return "-".join(
        part for part in (_language_token(language), _slug(script_variant), _slug(writing_mode)) if part
    )


def _default_training_backend(language: str, writing_mode: str) -> str:
    if writing_mode == "printed" and language == "syriac":
        return "tesstrain"
    return "kraken"


def _ssh_transport(port: int, ssh_key_path: Optional[Path]) -> str:
    command = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(port),
    ]
    if ssh_key_path is not None:
        command.extend(["-i", str(ssh_key_path)])
    return " ".join(shlex.quote(part) for part in command)


def _resolve_ssh_port(port_mappings: Dict[str, int]) -> int:
    for key in _SSH_PORT_KEYS:
        if key in port_mappings:
            return int(port_mappings[key])
    raise RuntimeError(
        "RunPod pod does not expose TCP port 22. Enable SSH/public IP access before retrieving artifacts."
    )


def recommend_gpu_tier(*, corpus_size: int, training_backend: str) -> str:
    """Choose a RunPod GPU tier using the Harness update policy."""

    backend = training_backend.strip().lower()
    if backend in {"tesstrain", "tesseract"}:
        return "rtx4090"
    if corpus_size > 20_000:
        return "a100"
    return "rtx4090"


def _gpu_type_ids_for_tier(gpu_tier: str) -> tuple[str, ...]:
    key = _slug(gpu_tier)
    if key not in GPU_TIER_CHOICES:
        raise ValueError(
            f"Unsupported RunPod GPU tier: {gpu_tier}. Supported: {sorted(GPU_TIER_CHOICES)}"
        )
    return GPU_TIER_CHOICES[key]


def _build_train_command(
    *,
    language: str,
    mode: str,
    manifest_id: str,
    config_path: Optional[Path],
    partition: str,
) -> str:
    command = [
        "uv",
        "run",
        "msocr",
        "train",
        "--lang",
        language,
        "--mode",
        mode,
        "--split-manifest-id",
        manifest_id,
        "--split-partition",
        partition,
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    return " ".join(shlex.quote(part) for part in command)


@dataclass(frozen=True)
class RunPodTrainingJob:
    """Pod creation request for a training container."""

    name: str
    image_name: str
    gpu_type_ids: tuple[str, ...]
    env: Dict[str, str]
    docker_start_cmd: tuple[str, ...]
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    compute_type: str = "GPU"
    container_disk_in_gb: int = 50
    volume_in_gb: int = 20
    volume_mount_path: str = "/workspace"
    interruptible: bool = False
    support_public_ip: bool = False
    ports: tuple[str, ...] = ()
    data_center_ids: tuple[str, ...] = ()
    network_volume_id: Optional[str] = None
    container_registry_auth_id: Optional[str] = None

    def to_api_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name[:191],
            "imageName": self.image_name,
            "cloudType": self.cloud_type,
            "computeType": self.compute_type,
            "gpuCount": self.gpu_count,
            "gpuTypeIds": list(self.gpu_type_ids),
            "gpuTypePriority": "custom",
            "containerDiskInGb": self.container_disk_in_gb,
            "volumeInGb": self.volume_in_gb,
            "volumeMountPath": self.volume_mount_path,
            "interruptible": self.interruptible,
            "supportPublicIp": self.support_public_ip,
            "env": self.env,
            "dockerStartCmd": list(self.docker_start_cmd),
        }
        if self.ports:
            payload["ports"] = list(self.ports)
        if self.data_center_ids:
            payload["dataCenterIds"] = list(self.data_center_ids)
            payload["dataCenterPriority"] = "custom"
        if self.network_volume_id:
            payload["networkVolumeId"] = self.network_volume_id
        if self.container_registry_auth_id:
            payload["containerRegistryAuthId"] = self.container_registry_auth_id
        return payload


@dataclass(frozen=True)
class RunPodModelRetrieval:
    """rsync-based model retrieval plan for a running RunPod pod."""

    pod_id: str
    public_ip: str
    ssh_port: int
    remote_path: str
    local_path: Path
    ssh_user: str = "root"
    ssh_key_path: Optional[Path] = None

    def to_rsync_command(self) -> list[str]:
        source = f"{self.ssh_user}@{self.public_ip}:{self.remote_path}"
        return [
            "rsync",
            "-avzP",
            "-e",
            _ssh_transport(self.ssh_port, self.ssh_key_path),
            source,
            str(self.local_path),
        ]


@dataclass(frozen=True)
class RunPodPod:
    """Normalized RunPod pod response."""

    pod_id: str
    name: str
    desired_status: str
    image: str
    public_ip: Optional[str]
    port_mappings: Dict[str, int]
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RunPodPod":
        return cls(
            pod_id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            desired_status=str(payload.get("desiredStatus", "UNKNOWN")),
            image=str(payload.get("image", payload.get("imageName", ""))),
            public_ip=payload.get("publicIp"),
            port_mappings={
                str(key): int(value)
                for key, value in dict(payload.get("portMappings") or {}).items()
            },
            raw=payload,
        )


class RunPodClient:
    """Thin client over the official RunPod REST pods API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = RUNPOD_BASE_URL,
        timeout_sec: int = 30,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    @classmethod
    def from_env(cls) -> "RunPodClient":
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            raise RuntimeError("RUNPOD_API_KEY is required to submit or poll RunPod jobs.")
        return cls(api_key=api_key)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API {method} {path} failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"RunPod API {method} {path} failed: {exc.reason}") from exc

        if not raw:
            return {}
        return json.loads(raw)

    def create_pod(self, job: RunPodTrainingJob) -> RunPodPod:
        payload = self._request("POST", "/pods", payload=job.to_api_payload())
        return RunPodPod.from_payload(payload)

    def get_pod(self, pod_id: str) -> RunPodPod:
        payload = self._request("GET", f"/pods/{pod_id}")
        return RunPodPod.from_payload(payload)

    def stop_pod(self, pod_id: str) -> RunPodPod:
        payload = self._request("POST", f"/pods/{pod_id}/stop")
        return RunPodPod.from_payload(payload)

    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}")

    def build_model_retrieval(
        self,
        *,
        pod: RunPodPod,
        remote_path: str,
        local_path: Path,
        ssh_key_path: Optional[Path],
    ) -> RunPodModelRetrieval:
        if not pod.public_ip:
            raise RuntimeError(
                "RunPod pod does not have a public IP. Enable SSH/public IP access before retrieving artifacts."
            )
        if ssh_key_path is not None and not ssh_key_path.exists():
            raise RuntimeError(f"RunPod SSH key not found: {ssh_key_path}")
        return RunPodModelRetrieval(
            pod_id=pod.pod_id,
            public_ip=pod.public_ip,
            ssh_port=_resolve_ssh_port(pod.port_mappings),
            remote_path=remote_path,
            local_path=local_path,
            ssh_key_path=ssh_key_path,
        )

    def retrieve_model(
        self,
        *,
        pod: RunPodPod,
        remote_path: str,
        local_path: Path,
        ssh_key_path: Optional[Path],
        timeout_sec: int = 1800,
        poll_interval_sec: int = 15,
    ) -> Path:
        if shutil.which("rsync") is None:
            raise RuntimeError("rsync is required to retrieve model artifacts from RunPod.")

        retrieval = self.build_model_retrieval(
            pod=pod,
            remote_path=remote_path,
            local_path=local_path,
            ssh_key_path=ssh_key_path,
        )
        retrieval.local_path.parent.mkdir(parents=True, exist_ok=True)

        deadline = time.time() + timeout_sec
        last_error: Optional[str] = None
        while True:
            completed = subprocess.run(
                retrieval.to_rsync_command(),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0 and retrieval.local_path.exists():
                return retrieval.local_path

            output = (completed.stderr or completed.stdout or "").strip()
            if output:
                last_error = output
            if time.time() >= deadline:
                detail = f" Last rsync output: {last_error}" if last_error else ""
                raise TimeoutError(
                    "Timed out waiting for RunPod model artifact at "
                    f"{remote_path} from pod {pod.pod_id}.{detail}"
                )
            time.sleep(poll_interval_sec)

    def wait_for_status(
        self,
        pod_id: str,
        *,
        target_statuses: Iterable[str] = ("RUNNING",),
        terminal_statuses: Iterable[str] = ("TERMINATED",),
        poll_interval_sec: int = 10,
        timeout_sec: int = 900,
    ) -> RunPodPod:
        targets = {status.upper() for status in target_statuses}
        terminals = {status.upper() for status in terminal_statuses}
        deadline = time.time() + timeout_sec

        while True:
            pod = self.get_pod(pod_id)
            status = pod.desired_status.upper()
            if status in targets:
                return pod
            if status in terminals:
                raise RuntimeError(
                    f"RunPod pod {pod_id} reached terminal status {status} before {sorted(targets)}."
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for RunPod pod {pod_id} to reach {sorted(targets)}."
                )
            time.sleep(poll_interval_sec)


def build_training_job(
    *,
    manifest: FrozenManifest,
    language: str,
    script_variant: str,
    writing_mode: str,
    mode: str,
    pipeline_run_id: str,
    model_version: str,
    training_image: str = DEFAULT_TRAINING_IMAGE,
    training_backend: Optional[str] = None,
    config_path: Optional[Path] = None,
    partition: str = "train",
    gpu_tier: str = "auto",
    command: Optional[str] = None,
    volume_in_gb: int = 20,
    container_disk_in_gb: int = 50,
    interruptible: bool = False,
    network_volume_id: Optional[str] = None,
    container_registry_auth_id: Optional[str] = None,
    data_center_ids: Iterable[str] = (),
    enable_ssh: bool = False,
    ssh_public_key: Optional[str] = None,
) -> RunPodTrainingJob:
    normalized_language = normalize_language_code(language)
    normalized_writing_mode = _slug(writing_mode)
    normalized_mode = mode.strip().lower()
    normalized_variant = _slug(script_variant)
    cases = manifest.get_partition(partition)
    corpus_size = len(cases)
    backend = _slug(training_backend or _default_training_backend(normalized_language, normalized_writing_mode))
    resolved_gpu_tier = (
        recommend_gpu_tier(corpus_size=corpus_size, training_backend=backend)
        if _slug(gpu_tier) == "auto"
        else _slug(gpu_tier)
    )
    command_text = command or _build_train_command(
        language=normalized_language,
        mode=normalized_mode,
        manifest_id=manifest.manifest_id,
        config_path=config_path,
        partition=partition,
    )
    artifact_token = _artifact_token(
        normalized_language,
        normalized_variant,
        normalized_writing_mode,
    )
    pod_name = f"msocr-{artifact_token}-{pipeline_run_id}"[:191]
    env = {
        "MSOCR_LANGUAGE": normalized_language,
        "MSOCR_SCRIPT_VARIANT": normalized_variant,
        "MSOCR_WRITING_MODE": normalized_writing_mode,
        "MSOCR_TRAINING_MODE": normalized_mode,
        "MSOCR_MANIFEST_ID": manifest.manifest_id,
        "MSOCR_PIPELINE_RUN_ID": pipeline_run_id,
        "MSOCR_MODEL_VERSION": model_version,
        "MSOCR_TRAINING_BACKEND": backend,
        "MSOCR_CORPUS_SIZE": str(corpus_size),
        "MSOCR_GPU_TIER": resolved_gpu_tier,
    }
    if config_path is not None:
        env["MSOCR_CONFIG_PATH"] = str(config_path)
    if enable_ssh and ssh_public_key:
        env["SSH_PUBLIC_KEY"] = ssh_public_key.strip()

    return RunPodTrainingJob(
        name=pod_name,
        image_name=training_image,
        gpu_type_ids=_gpu_type_ids_for_tier(resolved_gpu_tier),
        env=env,
        docker_start_cmd=("bash", "-lc", command_text),
        container_disk_in_gb=container_disk_in_gb,
        volume_in_gb=volume_in_gb,
        interruptible=interruptible,
        support_public_ip=enable_ssh,
        ports=("22/tcp",) if enable_ssh else (),
        network_volume_id=network_volume_id,
        container_registry_auth_id=container_registry_auth_id,
        data_center_ids=tuple(data_center_ids),
    )
