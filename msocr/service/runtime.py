"""Shared runtime helpers for CLI/API/demo execution paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from msocr.language_registry import normalize_language_code
from msocr.models.inference import predict
from msocr.pipeline.har_client import HARClient, build_model_artifact_name
from msocr.pipelines.printed_ocr import run_printed_ocr


DEFAULT_RUNTIME_MODEL_CACHE_DIR = Path("models/runtime")


def _env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _override_path(*env_names: str) -> Optional[Path]:
    raw_path = _env_value(*env_names)
    if not raw_path:
        return None
    model_path = Path(raw_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Configured runtime model path not found: {model_path}")
    return model_path


def _printed_runtime_override_path() -> Optional[Path]:
    return _override_path("MSOCR_PRINTED_RUNTIME_MODEL_PATH", "MSOCR_RUNTIME_MODEL_PATH")


def _htr_runtime_override_path() -> Optional[Path]:
    return _override_path("MSOCR_HTR_RUNTIME_MODEL_PATH")


def _runtime_har_settings(
    *,
    registry_env_names: tuple[str, ...],
    version_env_names: tuple[str, ...],
    filename_env_names: tuple[str, ...],
    package_env_names: tuple[str, ...],
    pkg_url_env_names: tuple[str, ...],
    cache_dir_env_names: tuple[str, ...],
) -> Optional[Dict[str, str]]:
    keys = {
        "registry": _env_value(*registry_env_names),
        "version": _env_value(*version_env_names),
        "filename": _env_value(*filename_env_names),
        "package": _env_value(*package_env_names),
        "pkg_url": _env_value(*pkg_url_env_names),
        "cache_dir": _env_value(*cache_dir_env_names),
    }
    if not any(keys.values()):
        return None
    if not keys["registry"] or not keys["version"] or not keys["filename"]:
        raise RuntimeError(
            "Runtime HAR model configuration requires MSOCR_RUNTIME_HAR_REGISTRY, "
            "MSOCR_RUNTIME_HAR_VERSION, and MSOCR_RUNTIME_HAR_FILENAME."
        )
    return keys


def _printed_runtime_har_settings() -> Optional[Dict[str, str]]:
    return _runtime_har_settings(
        registry_env_names=("MSOCR_PRINTED_RUNTIME_HAR_REGISTRY", "MSOCR_RUNTIME_HAR_REGISTRY"),
        version_env_names=("MSOCR_PRINTED_RUNTIME_HAR_VERSION", "MSOCR_RUNTIME_HAR_VERSION"),
        filename_env_names=("MSOCR_PRINTED_RUNTIME_HAR_FILENAME", "MSOCR_RUNTIME_HAR_FILENAME"),
        package_env_names=("MSOCR_PRINTED_RUNTIME_HAR_PACKAGE", "MSOCR_RUNTIME_HAR_PACKAGE"),
        pkg_url_env_names=("MSOCR_PRINTED_RUNTIME_HAR_PKG_URL", "MSOCR_RUNTIME_HAR_PKG_URL"),
        cache_dir_env_names=("MSOCR_PRINTED_RUNTIME_HAR_CACHE_DIR", "MSOCR_RUNTIME_HAR_CACHE_DIR"),
    )


def _htr_runtime_har_settings() -> Optional[Dict[str, str]]:
    return _runtime_har_settings(
        registry_env_names=("MSOCR_HTR_RUNTIME_HAR_REGISTRY",),
        version_env_names=("MSOCR_HTR_RUNTIME_HAR_VERSION",),
        filename_env_names=("MSOCR_HTR_RUNTIME_HAR_FILENAME",),
        package_env_names=("MSOCR_HTR_RUNTIME_HAR_PACKAGE",),
        pkg_url_env_names=("MSOCR_HTR_RUNTIME_HAR_PKG_URL",),
        cache_dir_env_names=("MSOCR_HTR_RUNTIME_HAR_CACHE_DIR", "MSOCR_RUNTIME_HAR_CACHE_DIR"),
    )


def _resolve_runtime_package_name(
    *,
    language: str,
    script_variant: str,
    writing_mode: str,
    explicit_package: str,
) -> str:
    if explicit_package:
        return explicit_package
    return build_model_artifact_name(language, script_variant, writing_mode)


def _runtime_cache_dir(settings: Dict[str, str]) -> Path:
    raw_path = settings.get("cache_dir", "").strip()
    return Path(raw_path) if raw_path else DEFAULT_RUNTIME_MODEL_CACHE_DIR


def _pull_runtime_model_from_har(
    *,
    language: str,
    script_variant: str,
    writing_mode: str,
    settings: Dict[str, str],
) -> Optional[Path]:
    package_name = _resolve_runtime_package_name(
        language=language,
        script_variant=script_variant,
        writing_mode=writing_mode,
        explicit_package=settings["package"],
    )
    destination = _runtime_cache_dir(settings) / Path(package_name) / settings["version"] / settings["filename"]
    client = HARClient(**({"pkg_url": settings["pkg_url"]} if settings["pkg_url"] else {}))
    return client.pull_file(
        registry=settings["registry"],
        package_name=package_name,
        version=settings["version"],
        filename=settings["filename"],
        destination=destination,
        pkg_url=settings["pkg_url"] or None,
    )


def resolve_printed_runtime_model_path(
    *,
    language: str,
    script_variant: str,
    model: Optional[str],
) -> Optional[str]:
    if model:
        return model
    override_path = _printed_runtime_override_path()
    if override_path is not None:
        return str(override_path)
    settings = _printed_runtime_har_settings()
    if settings is None:
        return None
    pulled_path = _pull_runtime_model_from_har(
        language=language,
        script_variant=script_variant,
        writing_mode="printed",
        settings=settings,
    )
    if pulled_path is None:
        return None
    return str(pulled_path)


def prefetch_printed_runtime_model_from_env() -> Optional[Path]:
    override_path = _printed_runtime_override_path()
    if override_path is not None:
        return override_path
    settings = _printed_runtime_har_settings()
    if settings is None:
        return None
    if settings["package"]:
        return _pull_runtime_model_from_har(
            language="syriac",
            script_variant="default",
            writing_mode="printed",
            settings=settings,
        )
    return None


def resolve_htr_runtime_model_path(
    *,
    language: str,
    script_variant: str,
    model: Optional[str],
) -> Optional[str]:
    if model:
        return model
    override_path = _htr_runtime_override_path()
    if override_path is not None:
        return str(override_path)
    settings = _htr_runtime_har_settings()
    if settings is None:
        return None
    pulled_path = _pull_runtime_model_from_har(
        language=language,
        script_variant=script_variant,
        writing_mode="handwritten",
        settings=settings,
    )
    if pulled_path is None:
        return None
    return str(pulled_path)


def prefetch_htr_runtime_model_from_env() -> Optional[Path]:
    override_path = _htr_runtime_override_path()
    if override_path is not None:
        return override_path
    settings = _htr_runtime_har_settings()
    if settings is None:
        return None

    language = _env_value(
        "MSOCR_HTR_RUNTIME_LANG",
        "MSOCR_HTR_RUNTIME_LANGUAGE",
        "MSOCR_HTR_RUNTIME_HAR_LANG",
    )
    script_variant = _env_value("MSOCR_HTR_RUNTIME_VARIANT") or "default"
    if not settings["package"] and not language:
        return None

    return _pull_runtime_model_from_har(
        language=language or "syriac",
        script_variant=script_variant,
        writing_mode="handwritten",
        settings=settings,
    )


def run_printed_service(
    *,
    lang: str,
    image_path: Path,
    model: Optional[str] = None,
    engine: str = "auto",
    variant: str = "default",
    reference_text_path: Optional[str] = None,
    cer_threshold: float = 0.05,
    device: str = "cpu",
) -> Dict[str, Any]:
    lang_key = normalize_language_code(lang)
    resolved_model = resolve_printed_runtime_model_path(
        language=lang_key,
        script_variant=variant,
        model=model,
    )
    result = run_printed_ocr(
        lang=lang_key,
        image_path=image_path,
        model=resolved_model,
        device=device,
        engine=engine,
        variant=variant,
        reference_text_path=reference_text_path,
        cer_threshold=cer_threshold,
    )
    return {
        "text": result["text"],
        "engine": result["engine"],
        "mode": "ocr",
        "language": lang_key,
    }


def run_htr_service(
    *,
    lang: str,
    image_path: Path,
    model: Optional[str] = None,
    provider: str = "auto",
    variant: str = "default",
    device: str = "cpu",
) -> Dict[str, Any]:
    lang_key = normalize_language_code(lang)
    provider_key = provider.strip().lower()
    resolved_model = resolve_htr_runtime_model_path(
        language=lang_key,
        script_variant=variant,
        model=model,
    )

    if lang_key == "syriac" and provider_key == "transkribus":
        return {
            "text": (
                "Syriac handwritten route is configured for Transkribus workflow currently. "
                "Export/import through Transkribus and then re-ingest results."
            ),
            "engine": "transkribus",
            "mode": "htr",
            "language": lang_key,
        }

    if lang_key == "syriac" and provider_key == "auto" and resolved_model is None:
        return {
            "text": (
                "Syriac handwritten route is configured for Transkribus workflow currently. "
                "Export/import through Transkribus and then re-ingest results."
            ),
            "engine": "transkribus",
            "mode": "htr",
            "language": lang_key,
        }

    if resolved_model:
        model_path = Path(resolved_model)
    elif lang_key == "latin":
        model_path = Path("models/kraken/latin_handwritten_mccatmus.mlmodel")
    elif lang_key == "greek":
        model_path = Path(
            "models/kraken/greek-german_serifs_sophokle1v3soph/"
            "greek-german_serifs_sophokle1v3soph.mlmodel"
        )
    else:
        raise ValueError("HTR model is required for this language.")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    text = predict(str(image_path), str(model_path), device=device)
    return {
        "text": text,
        "engine": "kraken",
        "mode": "htr",
        "language": lang_key,
    }
