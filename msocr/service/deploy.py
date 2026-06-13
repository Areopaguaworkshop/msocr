"""Deployment-oriented startup and smoke-check helpers for the HTR service."""

from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

import uvicorn

from msocr.language_registry import normalize_language_code
from msocr.service.runtime import resolve_htr_runtime_model_path, run_htr_service


def _htr_runtime_source(model: Optional[str]) -> str:
    if model:
        return "explicit"
    if os.getenv("MSOCR_HTR_RUNTIME_MODEL_PATH", "").strip():
        return "env_path"
    if os.getenv("MSOCR_HTR_MODEL_PATH", "").strip():
        return "env_path"
    if os.getenv("MSOCR_RUNTIME_MODEL_PATH", "").strip():
        return "env_path"
    return "default_route"


def _build_runtime_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _decode_json_response(response: Any) -> Dict[str, Any]:
    raw = response.read().decode("utf-8")
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Runtime smoke endpoint returned a non-object JSON payload.")
    return payload


def _build_multipart_request(
    *,
    fields: Dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"msocr-runtime-smoke-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def wait_for_runtime_health(
    *,
    base_url: str,
    timeout_sec: int = 120,
    poll_interval_sec: int = 3,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    health_url = _build_runtime_url(base_url, "/health")
    last_error: Optional[str] = None

    while True:
        try:
            req = request.Request(health_url, method="GET")
            with request.urlopen(req, timeout=min(timeout_sec, 10)) as response:
                return _decode_json_response(response)
        except (error.HTTPError, error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = str(exc)

        if time.time() >= deadline:
            detail = f" Last error: {last_error}" if last_error else ""
            raise TimeoutError(
                f"Timed out waiting for runtime health endpoint at {health_url}.{detail}"
            )
        time.sleep(poll_interval_sec)


def runtime_htr_smoke_check(
    *,
    language: str,
    script_variant: str,
    model: Optional[str] = None,
    image_path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    normalized_language = normalize_language_code(language)
    resolved_model = resolve_htr_runtime_model_path(
        language=normalized_language,
        script_variant=script_variant,
        model=model,
    )
    payload: Dict[str, Any] = {
        "ok": True,
        "mode": "htr",
        "writing_mode": "handwritten",
        "language": normalized_language,
        "script_variant": script_variant,
        "runtime_source": _htr_runtime_source(model),
        "model_path": resolved_model,
        "image_path": str(image_path) if image_path is not None else None,
    }
    if image_path is None:
        return payload

    if not image_path.exists():
        raise FileNotFoundError(f"Runtime smoke image not found: {image_path}")

    result = run_htr_service(
        lang=normalized_language,
        image_path=image_path,
        model=model,
        variant=script_variant,
        device=device,
    )
    payload["engine"] = result["engine"]
    payload["text_length"] = len(result["text"])
    return payload


def runtime_http_htr_smoke_check(
    *,
    base_url: str,
    language: str,
    script_variant: str,
    image_path: Optional[Path],
    device: str = "cpu",
    timeout_sec: int = 120,
    poll_interval_sec: int = 3,
) -> Dict[str, Any]:
    if image_path is None:
        raise ValueError("HTR runtime HTTP smoke check requires an image path.")
    normalized_language = normalize_language_code(language)
    if not image_path.exists():
        raise FileNotFoundError(f"Runtime smoke image not found: {image_path}")

    health_payload = wait_for_runtime_health(
        base_url=base_url,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )

    body, content_type = _build_multipart_request(
        fields={
            "lang": normalized_language,
            "variant": script_variant,
            "device": device,
        },
        file_field="file",
        file_path=image_path,
    )
    htr_url = _build_runtime_url(base_url, "/htr")
    req = request.Request(
        htr_url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            payload = _decode_json_response(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runtime HTR smoke request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Runtime HTR smoke request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Runtime HTR smoke request returned invalid JSON.") from exc

    text = str(payload.get("text", "")).strip()
    if not payload.get("ok"):
        raise RuntimeError(f"Runtime HTR smoke request failed: {payload}")
    if not text:
        raise RuntimeError("Runtime HTR smoke request returned empty text.")

    return {
        "ok": True,
        "base_url": base_url.rstrip("/"),
        "health": health_payload,
        "htr": {
            "engine": payload.get("engine"),
            "language": payload.get("language", normalized_language),
            "mode": payload.get("mode", "htr"),
            "image_path": str(image_path),
            "text_length": len(text),
            "text_preview": text[:80],
        },
    }


def gradio_http_smoke_check(
    *,
    base_url: str,
    expected_text: str = "msocr Sogdian HTR",
    timeout_sec: int = 120,
    poll_interval_sec: int = 3,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    root_url = _build_runtime_url(base_url, "/")
    last_error: Optional[str] = None

    while True:
        try:
            req = request.Request(root_url, method="GET")
            with request.urlopen(req, timeout=min(timeout_sec, 10)) as response:
                html = response.read().decode("utf-8", errors="replace")
                if expected_text not in html:
                    raise RuntimeError(
                        f"Gradio root did not include expected text {expected_text!r}."
                    )
                return {
                    "ok": True,
                    "base_url": base_url.rstrip("/"),
                    "expected_text": expected_text,
                }
        except (error.HTTPError, error.URLError, RuntimeError) as exc:
            last_error = str(exc)

        if time.time() >= deadline:
            detail = f" Last error: {last_error}" if last_error else ""
            raise TimeoutError(f"Timed out waiting for Gradio at {root_url}.{detail}")
        time.sleep(poll_interval_sec)


def preflight_runtime_from_env() -> Dict[str, Any]:
    language = os.getenv("MSOCR_RUNTIME_SMOKE_LANG", "sogdian").strip() or "sogdian"
    script_variant = os.getenv("MSOCR_RUNTIME_SMOKE_VARIANT", "standard").strip() or "standard"
    image_path_raw = os.getenv("MSOCR_RUNTIME_SMOKE_IMAGE", "").strip()
    image_path = Path(image_path_raw) if image_path_raw else None
    device = os.getenv("MSOCR_RUNTIME_SMOKE_DEVICE", "cpu").strip() or "cpu"
    return runtime_htr_smoke_check(
        language=language,
        script_variant=script_variant,
        image_path=image_path,
        device=device,
    )


def run_api_server(*, host: str, port: int, reload: bool) -> None:
    preflight_runtime_from_env()
    uvicorn.run("msocr.service.api:app", host=host, port=port, reload=reload)


def run_demo_server(*, host: str, port: int, share: bool) -> None:
    preflight_runtime_from_env()
    from msocr.service.gradio_demo import build_demo

    demo = build_demo()
    demo.launch(server_name=host, server_port=port, share=share)
