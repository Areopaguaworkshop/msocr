"""Tests for printed OCR routing overrides."""

from pathlib import Path

from msocr.pipelines.printed_ocr import run_printed_ocr


def test_syriac_tesseract_route_uses_explicit_model_file(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "page.png"
    model_path = tmp_path / "custom_syr.traineddata"
    image_path.write_bytes(b"png")
    model_path.write_text("model", encoding="utf-8")

    seen = {}

    def fake_run_tesseract(image_arg, lang_arg, tessdata_dir=None):
        seen["image"] = image_arg
        seen["lang"] = lang_arg
        seen["tessdata_dir"] = tessdata_dir
        return "ܫܠܡܐ"

    monkeypatch.setattr("msocr.pipelines.printed_ocr._run_tesseract", fake_run_tesseract)

    result = run_printed_ocr(
        lang="syriac",
        image_path=image_path,
        model=str(model_path),
        device="cpu",
        engine="tesseract",
        variant="estrangela",
    )

    assert result["engine"] == "tesseract"
    assert seen["image"] == image_path
    assert seen["lang"] == "custom_syr"
    assert seen["tessdata_dir"] == model_path.parent
