"""Printed OCR routing pipeline.

Greek printed OCR uses ocrmypdf (Tesseract engine) as default.
Latin printed OCR uses Kraken CATMuS-Print Large (2024-01-30, 98.56% accuracy, CER 1.44%)
  with Tesseract fallback. Also supports CATMuS Medieval for handwritten Latin.
Syriac printed OCR uses Tesseract (Estrangela baseline), with optional
CER-gated switch to custom Serto/East Syriac Tesseract models.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, TypedDict

from click import ClickException

from msocr.language_registry import normalize_language_code
from msocr.models.inference import predict


class PrintedOCRResult(TypedDict):
    text: str
    engine: str
    language: str


# CATMuS-Print Large (2024-01-30): Latest model for printed Latin (98.56% accuracy, CER 1.44%)
# DOI: 10.5281/zenodo.10592716
LATIN_PRIMARY_MODEL = Path("models/kraken/catmus-print-fondue-large.mlmodel")
# CATMuS Medieval: For handwritten Latin manuscripts (8-15th century)
# DOI: 10.5281/zenodo.10066218
LATIN_SECONDARY_MODEL = Path("models/kraken/catmus-medieval-1.5.0.mlmodel")
SYRIAC_TESSDATA_DIR = Path("models/tesseract")
SYRIAC_SERTO_LANG = "syr_serto"
SYRIAC_EAST_LANG = "syr_east"
COPTIC_TESSDATA_DIR = Path("models/tesseract")
COPTIC_LOCAL_MODEL = COPTIC_TESSDATA_DIR / "cop.traineddata"
ARMENIAN_TESSDATA_DIR = Path("models/tesseract")
ARMENIAN_LOCAL_MODEL = ARMENIAN_TESSDATA_DIR / "hye-calfa-n.traineddata"


def _run_ocrmypdf(
    image_path: Path,
    lang: str = "grc",
) -> str:
    """Run ocrmypdf for OCR on images/PDFs using Tesseract engine."""
    if shutil.which("ocrmypdf") is None:
        raise ClickException(
            "ocrmypdf not found on PATH. Install ocrmypdf to run Greek OCR: pip install ocrmypdf"
        )

    from PIL import Image

    temp_image_path: Optional[Path] = None
    try:
        with Image.open(image_path) as img:
            if img.mode == "RGBA" or img.mode == "LA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                temp_image_path = image_path.with_suffix(".png")
                rgb_img.save(temp_image_path)
                image_path = temp_image_path
            elif img.mode != "RGB":
                temp_image_path = image_path.with_suffix(".png")
                img.convert("RGB").save(temp_image_path)
                image_path = temp_image_path
    except Exception as e:
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()
        raise ClickException(f"Failed to preprocess image: {e}") from e

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ocrmypdf",
            "--sidecar", str(tmp_path),
            "-l", lang,
            "--image-dpi", "300",
            "--skip-text",
            str(image_path),
            "/dev/null",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        error_output = exc.stderr.strip() if exc.stderr else str(exc)
        raise ClickException(f"ocrmypdf failed: {error_output}") from exc

    try:
        text = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()

    return text.strip()


def _run_tesseract(
    image_path: Path,
    tesseract_lang: str,
    tessdata_dir: Optional[Path] = None,
) -> str:
    if shutil.which("tesseract") is None:
        raise ClickException(
            "Tesseract binary not found on PATH. Install tesseract OCR to run Latin printed OCR."
        )

    cmd = ["tesseract", str(image_path), "stdout", "-l", tesseract_lang]
    if tessdata_dir:
        cmd.extend(["--tessdata-dir", str(tessdata_dir)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr.strip() if exc.stderr else str(exc)
        raise ClickException(f"Tesseract OCR failed: {error_output}") from exc
    return result.stdout.strip()


def _run_tesseract_with_lang_fallback(
    image_path: Path,
    lang_candidates: list[str],
    tessdata_dir: Optional[Path] = None,
) -> str:
    last_error: Optional[Exception] = None
    for lang in lang_candidates:
        try:
            return _run_tesseract(image_path, lang, tessdata_dir=tessdata_dir)
        except ClickException as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ClickException("No usable Tesseract language candidates were provided.")


def _run_kraken_with_fallback(image_path: Path, model_path: Path, device: str, segmentation_type: str = "baseline", min_length: int = 10) -> str:
    text = predict(str(image_path), str(model_path), device=device, segmentation_type=segmentation_type).strip()
    if text and not text.startswith("Error:") and len(text) >= min_length:
        return text
    return ""


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _char_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 1.0 if hyp else 0.0
    return _levenshtein(ref, hyp) / len(ref)


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    text_path = Path(path)
    if not text_path.exists():
        raise ClickException(f"Reference text file not found: {text_path}")
    return text_path.read_text(encoding="utf-8")


def run_printed_ocr(
    *,
    lang: str,
    image_path: Path,
    model: Optional[str],
    device: str,
    engine: str,
    variant: str = "default",
    reference_text_path: Optional[str] = None,
    cer_threshold: float = 0.05,
) -> PrintedOCRResult:
    try:
        lang_key = normalize_language_code(lang)
    except KeyError:
        lang_key = lang.lower()
    engine_key = engine.lower()
    variant_key = variant.lower()

    if lang_key == "latin":
        if engine_key == "auto":
            # Default to CATMuS-Print Large via Kraken; fallback to Tesseract.
            if LATIN_PRIMARY_MODEL.exists():
                text = _run_kraken_with_fallback(
                    image_path, LATIN_PRIMARY_MODEL, device
                )
                if text:
                    return {"text": text, "engine": "kraken", "language": lang_key}
            text = _run_tesseract(image_path, "lat")
            return {"text": text, "engine": "tesseract", "language": lang_key}
        if engine_key == "tesseract":
            text = _run_tesseract(image_path, "lat")
            return {"text": text, "engine": "tesseract", "language": lang_key}
        if engine_key == "kraken":
            model_path = Path(model) if model else LATIN_PRIMARY_MODEL
            if not model_path.exists():
                raise ClickException(
                    "Latin Kraken OCR model not found. Pass --model or place "
                    "models/kraken/latin_printed_catmus_large.mlmodel."
                )
            text = _run_kraken_with_fallback(image_path, model_path, device)
            if not text:
                raise ClickException("Latin Kraken OCR returned empty output.")
            return {"text": text, "engine": "kraken", "language": lang_key}
        raise ClickException(
            "Unsupported engine for Latin. Use auto, tesseract, or kraken."
        )

    if lang_key == "greek":
        if engine_key == "auto":
            # Default to ocrmypdf (Tesseract engine) for Greek
            text = _run_ocrmypdf(image_path, lang="grc")
            if text:
                return {"text": text, "engine": "ocrmypdf", "language": lang_key}
            raise ClickException(
                "Greek OCR returned empty output with ocrmypdf."
            )

        if engine_key == "ocrmypdf":
            text = _run_ocrmypdf(image_path, lang="grc")
            if text:
                return {"text": text, "engine": "ocrmypdf", "language": lang_key}
            raise ClickException("Greek ocrmypdf OCR returned empty output.")

        if engine_key == "tesseract":
            text = _run_tesseract(image_path, "grc")
            if text:
                return {"text": text, "engine": "tesseract", "language": lang_key}
            raise ClickException("Greek Tesseract OCR returned empty output.")

        raise ClickException("Unsupported engine for Greek. Use auto, ocrmypdf, or tesseract.")

    if lang_key == "syriac":
        if engine_key == "kraken":
            raise ClickException(
                "Syriac printed OCR is configured for Tesseract in this pipeline. "
                "Use --engine tesseract or auto."
            )
        if engine_key not in ("auto", "tesseract"):
            raise ClickException(
                "Unsupported engine for Syriac. Use auto or tesseract."
            )

        base_text = _run_tesseract(image_path, "syr")
        # Estrangela/default stays on baseline Tesseract model.
        if variant_key in ("default", "estrangela"):
            return {"text": base_text, "engine": "tesseract", "language": lang_key}

        if variant_key not in ("serto", "east"):
            raise ClickException(
                "Unsupported Syriac variant. Use default, estrangela, serto, or east."
            )

        reference_text = _read_text_file(reference_text_path)
        if reference_text is None:
            # No CER gate can be computed without a reference.
            return {"text": base_text, "engine": "tesseract", "language": lang_key}

        cer = _char_error_rate(reference_text, base_text)
        if cer <= cer_threshold:
            return {"text": base_text, "engine": "tesseract", "language": lang_key}

        # CER gate failed; attempt custom Serto/East model if available.
        custom_lang = SYRIAC_SERTO_LANG if variant_key == "serto" else SYRIAC_EAST_LANG
        custom_file = SYRIAC_TESSDATA_DIR / f"{custom_lang}.traineddata"
        if not custom_file.exists():
            return {"text": base_text, "engine": "tesseract", "language": lang_key}

        custom_text = _run_tesseract(
            image_path,
            custom_lang,
            tessdata_dir=SYRIAC_TESSDATA_DIR,
        )
        return {"text": custom_text, "engine": "tesseract", "language": lang_key}

    if lang_key == "coptic":
        if engine_key == "kraken":
            raise ClickException(
                "Coptic printed OCR is configured for Tesseract in this pipeline. "
                "Use --engine tesseract or auto."
            )
        if engine_key not in ("auto", "tesseract"):
            raise ClickException(
                "Unsupported engine for Coptic. Use auto or tesseract."
            )

        # Tesseract path
        if COPTIC_LOCAL_MODEL.exists():
            text = _run_tesseract(image_path, "cop", tessdata_dir=COPTIC_TESSDATA_DIR)
        else:
            text = _run_tesseract(image_path, "cop")

        return {"text": text, "engine": "tesseract", "language": lang_key}

    if lang_key == "armenian":
        if engine_key in ("auto", "tesseract"):
            # Prefer local hye-calfa-n model, fallback to system hye.
            if ARMENIAN_LOCAL_MODEL.exists():
                try:
                    text = _run_tesseract(
                        image_path,
                        "hye-calfa-n",
                        tessdata_dir=ARMENIAN_TESSDATA_DIR,
                    )
                    return {"text": text, "engine": "tesseract", "language": lang_key}
                except ClickException:
                    pass
            text = _run_tesseract(image_path, "hye")
            return {"text": text, "engine": "tesseract", "language": lang_key}
        raise ClickException(
            "Armenian printed OCR currently supports Tesseract route in this pipeline."
        )

    if lang_key == "geez":
        if engine_key in ("auto", "tesseract"):
            # Prefer classical Geez code if available, then practical fallbacks.
            text = _run_tesseract_with_lang_fallback(
                image_path,
                ["gez", "tir", "amh"],
            )
            return {"text": text, "engine": "tesseract", "language": lang_key}
        raise ClickException(
            "Geez printed OCR currently supports Tesseract route in this pipeline."
        )

    raise ClickException(
        "Printed OCR pipeline currently implemented for Greek, Latin, Syriac, Coptic, Armenian, and Geez only. "
        f"Got: {lang_key}"
    )
