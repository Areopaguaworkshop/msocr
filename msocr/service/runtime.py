"""Shared runtime helpers for CLI/API/demo execution paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from msocr.models.inference import predict
from msocr.pipelines.printed_ocr import run_printed_ocr


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
    result = run_printed_ocr(
        lang=lang.strip().lower(),
        image_path=image_path,
        model=model,
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
        "language": lang.strip().lower(),
    }


def run_htr_service(
    *,
    lang: str,
    image_path: Path,
    model: Optional[str] = None,
    provider: str = "auto",
    device: str = "cpu",
) -> Dict[str, Any]:
    lang_key = lang.strip().lower()
    provider_key = provider.strip().lower()

    if lang_key == "syriac" and provider_key in ("auto", "transkribus"):
        return {
            "text": (
                "Syriac handwritten route is configured for Transkribus workflow currently. "
                "Export/import through Transkribus and then re-ingest results."
            ),
            "engine": "transkribus",
            "mode": "htr",
            "language": lang_key,
        }

    if model:
        model_path = Path(model)
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
