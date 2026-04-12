"""Tests for runtime model resolution helpers."""

from pathlib import Path


def test_run_printed_service_uses_har_runtime_model(monkeypatch, tmp_path: Path):
    from msocr.service.runtime import run_printed_service

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    pulled_model = tmp_path / "runtime" / "syr-estrangela-printed" / "v14" / "model.traineddata"

    def fake_pull_file(self, **kwargs):
        assert kwargs["registry"] == "msocr-models"
        assert kwargs["package_name"] == "syr-estrangela-printed"
        assert kwargs["version"] == "v14"
        assert kwargs["filename"] == "model.traineddata"
        pulled_model.parent.mkdir(parents=True, exist_ok=True)
        pulled_model.write_text("model", encoding="utf-8")
        return pulled_model

    observed = {}

    def fake_run_printed_ocr(**kwargs):
        observed["model"] = kwargs["model"]
        return {"text": "ܫܠܡܐ", "engine": "tesseract", "language": kwargs["lang"]}

    monkeypatch.setenv("MSOCR_RUNTIME_HAR_REGISTRY", "msocr-models")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_VERSION", "v14")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_FILENAME", "model.traineddata")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_CACHE_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr("msocr.service.runtime.HARClient.pull_file", fake_pull_file)
    monkeypatch.setattr("msocr.service.runtime.run_printed_ocr", fake_run_printed_ocr)

    result = run_printed_service(
        lang="syriac",
        image_path=image_path,
        engine="tesseract",
        variant="estrangela",
    )

    assert observed["model"] == str(pulled_model)
    assert result["text"] == "ܫܠܡܐ"


def test_prefetch_runtime_model_from_env_uses_explicit_package(monkeypatch, tmp_path: Path):
    from msocr.service.runtime import prefetch_printed_runtime_model_from_env

    pulled_model = tmp_path / "runtime" / "deploy" / "v14" / "model.traineddata"

    def fake_pull_file(self, **kwargs):
        assert kwargs["package_name"] == "deploy/syriac-runtime"
        pulled_model.parent.mkdir(parents=True, exist_ok=True)
        pulled_model.write_text("model", encoding="utf-8")
        return pulled_model

    monkeypatch.setenv("MSOCR_RUNTIME_HAR_REGISTRY", "msocr-models")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_VERSION", "v14")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_FILENAME", "model.traineddata")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_PACKAGE", "deploy/syriac-runtime")
    monkeypatch.setenv("MSOCR_RUNTIME_HAR_CACHE_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr("msocr.service.runtime.HARClient.pull_file", fake_pull_file)

    resolved = prefetch_printed_runtime_model_from_env()

    assert resolved == pulled_model
