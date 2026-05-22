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
        harness_api_token: Optional[str] = None,
        account_id: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self.executable = executable
        self.pkg_url = pkg_url.rstrip("/")
        self.harness_api_token = (
            harness_api_token
            or harness_api_key
            or os.getenv("HARNESS_API_TOKEN")
            or os.getenv("HARNESS_API_KEY")
        )
        self.account_id = account_id or os.getenv("HARNESS_ACCOUNT_ID")
        self.org_id = org_id or os.getenv("HARNESS_ORG_ID")
        self.project_id = project_id or os.getenv("HARNESS_PROJECT_ID")
        self.api_url = api_url or os.getenv("HARNESS_API_URL")

    def _command_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.harness_api_token:
            env["HARNESS_API_TOKEN"] = self.harness_api_token
            env.setdefault("HARNESS_API_KEY", self.harness_api_token)
        return env

    def _ensure_cli(self) -> None:
        if shutil.which(self.executable) is None:
            raise RuntimeError(
                f"Harness CLI executable not found: {self.executable}. Install `hc` to publish HAR artifacts."
            )

    def _context_flags(self) -> list[str]:
        flags: list[str] = []
        if self.api_url:
            flags.extend(["--api-url", self.api_url])
        if self.account_id:
            flags.extend(["--account", self.account_id])
        if self.org_id:
            flags.extend(["--org", self.org_id])
        if self.project_id:
            flags.extend(["--project", self.project_id])
        return flags

    def build_login_command(self, *, redacted: bool = False) -> Optional[list[str]]:
        if not self.harness_api_token:
            return None
        token = "$HARNESS_API_TOKEN" if redacted else self.harness_api_token
        return [
            self.executable,
            "auth",
            "login",
            "--api-token",
            token,
            *self._context_flags(),
            "--non-interactive",
        ]

    def _authenticate_if_configured(self, env: Dict[str, str]) -> None:
        command = self.build_login_command(redacted=False)
        if command is None:
            return
        try:
            subprocess.run(command, check=True, env=env, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            message = "Harness CLI authentication failed before artifact publication."
            if detail:
                message = f"{message} {detail}"
            raise RuntimeError(message) from exc

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

    def build_pull_command(
        self,
        *,
        registry: str,
        package_name: str,
        version: str,
        filename: str,
        destination: Path,
        pkg_url: Optional[str] = None,
    ) -> list[str]:
        package_path = f"{package_name}/{version}/{filename}"
        return [
            self.executable,
            "artifact",
            "pull",
            "generic",
            registry,
            package_path,
            str(destination),
            "--pkg-url",
            (pkg_url or self.pkg_url).rstrip("/"),
        ]

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
        commands: list[list[str]] = []
        login_command = self.build_login_command(redacted=True)
        if login_command is not None:
            commands.append(login_command)
        commands.extend(self.build_push_command(bundle, upload) for upload in bundle.files)
        metadata_command = self.build_metadata_command(bundle)
        if metadata_command is not None:
            commands.append(metadata_command)
        return commands

    def publish_bundle(self, bundle: HARArtifactBundle) -> None:
        self._ensure_cli()
        env = self._command_env()
        self._authenticate_if_configured(env)
        commands = [self.build_push_command(bundle, upload) for upload in bundle.files]
        metadata_command = self.build_metadata_command(bundle)
        if metadata_command is not None:
            commands.append(metadata_command)
        for command in commands:
            try:
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Harness artifact command failed: {' '.join(command)}"
                ) from exc

    def pull_file(
        self,
        *,
        registry: str,
        package_name: str,
        version: str,
        filename: str,
        destination: Path,
        pkg_url: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        if destination.exists() and not force:
            return destination

        self._ensure_cli()
        destination.parent.mkdir(parents=True, exist_ok=True)
        env = self._command_env()
        self._authenticate_if_configured(env)
        command = self.build_pull_command(
            registry=registry,
            package_name=package_name,
            version=version,
            filename=filename,
            destination=destination,
            pkg_url=pkg_url,
        )
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Harness artifact pull command failed: {' '.join(command)}"
            ) from exc
        if not destination.exists():
            raise RuntimeError(
                f"Harness artifact pull did not produce the expected file: {destination}"
            )
        return destination


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
