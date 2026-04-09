"""Harness Artifact Registry adapter for model bundles and sidecars."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from msocr.language_registry import LANGUAGE_REGISTRY, normalize_language_code


DEFAULT_PKG_URL = "https://pkg.harness.io"


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def build_model_artifact_name(language: str, script_variant: str, writing_mode: str) -> str:
    normalized_language = normalize_language_code(language)
    metadata = LANGUAGE_REGISTRY.get(normalized_language, {})
    language_token = str(metadata.get("iso", normalized_language)).lower()
    return "-".join(
        part for part in (language_token, _slug(script_variant), _slug(writing_mode)) if part
    )


@dataclass(frozen=True)
class HARFileUpload:
    """Single file upload entry inside a generic package version."""

    source_path: Path
    filename: str
    package_path: str


@dataclass(frozen=True)
class HARArtifactBundle:
    """Logical group of model and sidecar files for a single HAR version."""

    registry: str
    package_name: str
    version: str
    pkg_url: str = DEFAULT_PKG_URL
    description: Optional[str] = None
    files: tuple[HARFileUpload, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def artifact_ref(self) -> str:
        return f"{self.package_name}:{self.version}"


class HARClient:
    """Wrapper around the official Harness CLI artifact workflow."""

    def __init__(
        self,
        *,
        executable: str = "hc",
        pkg_url: str = DEFAULT_PKG_URL,
        harness_api_key: Optional[str] = None,
    ) -> None:
        self.executable = executable
        self.pkg_url = pkg_url.rstrip("/")
        self.harness_api_key = harness_api_key or os.getenv("HARNESS_API_KEY")

    def _command_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.harness_api_key:
            env["HARNESS_API_KEY"] = self.harness_api_key
        return env

    def _ensure_cli(self) -> None:
        if shutil.which(self.executable) is None:
            raise RuntimeError(
                f"Harness CLI executable not found: {self.executable}. Install `hc` to publish HAR artifacts."
            )

    def build_push_command(self, bundle: HARArtifactBundle, upload: HARFileUpload) -> list[str]:
        command = [
            self.executable,
            "artifact",
            "push",
            "generic",
            bundle.registry,
            str(upload.source_path),
            "--name",
            bundle.package_name,
            "--version",
            bundle.version,
            "--filename",
            upload.filename,
            "--path",
            upload.package_path,
            "--pkg-url",
            bundle.pkg_url or self.pkg_url,
        ]
        if bundle.description:
            command.extend(["--description", bundle.description])
        return command

    def build_metadata_command(self, bundle: HARArtifactBundle) -> Optional[list[str]]:
        if not bundle.metadata:
            return None
        metadata_string = ",".join(f"{key}:{value}" for key, value in sorted(bundle.metadata.items()))
        return [
            self.executable,
            "artifact",
            "metadata",
            "set",
            "--registry",
            bundle.registry,
            "--package",
            bundle.package_name,
            "--version",
            bundle.version,
            "--metadata",
            metadata_string,
        ]

    def plan_commands(self, bundle: HARArtifactBundle) -> list[list[str]]:
        commands = [self.build_push_command(bundle, upload) for upload in bundle.files]
        metadata_command = self.build_metadata_command(bundle)
        if metadata_command is not None:
            commands.append(metadata_command)
        return commands

    def publish_bundle(self, bundle: HARArtifactBundle) -> None:
        self._ensure_cli()
        env = self._command_env()
        for command in self.plan_commands(bundle):
            try:
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Harness artifact command failed: {' '.join(command)}"
                ) from exc


def build_bundle(
    *,
    registry: str,
    language: str,
    script_variant: str,
    writing_mode: str,
    version: str,
    model_file: Path,
    metrics_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
    dockerfile_sha_file: Optional[Path] = None,
    pkg_url: str = DEFAULT_PKG_URL,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> HARArtifactBundle:
    package_name = build_model_artifact_name(language, script_variant, writing_mode)
    uploads = [
        HARFileUpload(
            source_path=model_file,
            filename=model_file.name,
            package_path=model_file.name,
        )
    ]
    for sidecar_path in (metrics_file, config_file, dockerfile_sha_file):
        if sidecar_path is None:
            continue
        uploads.append(
            HARFileUpload(
                source_path=sidecar_path,
                filename=sidecar_path.name,
                package_path=f"sidecars/{sidecar_path.name}",
            )
        )

    return HARArtifactBundle(
        registry=registry,
        package_name=package_name,
        version=version,
        pkg_url=pkg_url,
        description=description,
        files=tuple(uploads),
        metadata=metadata or {},
    )
