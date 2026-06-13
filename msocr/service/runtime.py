"""Shared local HTR runtime helpers for CLI/API/demo execution paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from msocr.language_registry import normalize_language_code
from msocr.models.inference import predict


DEFAULT_HTR_MODELS: dict[str, Path] = {
    "sogdian": Path("models/kraken/sogdian_manuscript.mlmodel"),
}


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
        raise FileNotFoundError(f"Configured HTR model path not found: {model_path}")
    return model_path


def resolve_htr_runtime_model_path(
    *,
    language: str,
    script_variant: str,
    model: Optional[str],
) -> Optional[str]:
    """Resolve the Kraken model for manuscript HTR.

    The runtime is intentionally local-only. It accepts an explicit model path,
    then environment overrides, then the checked-in default model location for
    the language. The script variant is kept in the signature because CLI/API
    callers record it, but model selection is not delegated to an external
    workflow layer anymore.
    """
    _ = script_variant
    if model:
        return model

    override_path = _override_path(
        "MSOCR_HTR_RUNTIME_MODEL_PATH",
        "MSOCR_HTR_MODEL_PATH",
        "MSOCR_RUNTIME_MODEL_PATH",
    )
    if override_path is not None:
        return str(override_path)

    lang_key = normalize_language_code(language)
    default_model = DEFAULT_HTR_MODELS.get(lang_key)
    return str(default_model) if default_model is not None else None


def prefetch_htr_runtime_model_from_env() -> Optional[Path]:
    """Validate and return the configured local HTR model path, if any."""
    return _override_path(
        "MSOCR_HTR_RUNTIME_MODEL_PATH",
        "MSOCR_HTR_MODEL_PATH",
        "MSOCR_RUNTIME_MODEL_PATH",
    )


def run_htr_service(
    *,
    lang: str,
    image_path: Path,
    model: Optional[str] = None,
    variant: str = "standard",
    device: str = "cpu",
) -> Dict[str, Any]:
    lang_key = normalize_language_code(lang)
    resolved_model = resolve_htr_runtime_model_path(
        language=lang_key,
        script_variant=variant,
        model=model,
    )
    if resolved_model is None:
        raise ValueError(f"HTR model is required for language: {lang_key}")

    model_path = Path(resolved_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    text = predict(str(image_path), str(model_path), device=device)
    return {
        "text": text,
        "engine": "kraken",
        "mode": "htr",
        "writing_mode": "handwritten",
        "language": lang_key,
        "variant": variant,
        "model_path": str(model_path),
    }
