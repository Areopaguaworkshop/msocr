"""Tests for local Sogdian HTR runtime helpers."""

from pathlib import Path

import pytest


def test_resolve_htr_runtime_model_prefers_explicit_model(tmp_path: Path):
    from msocr.service.runtime import resolve_htr_runtime_model_path

    model_path = tmp_path / "explicit.mlmodel"
    model_path.write_text("model", encoding="utf-8")

    assert (
        resolve_htr_runtime_model_path(
            language="sogdian",
            script_variant="standard",
            model=str(model_path),
        )
        == str(model_path)
    )


def test_prefetch_htr_runtime_model_from_env_validates_local_path(monkeypatch, tmp_path: Path):
    from msocr.service.runtime import prefetch_htr_runtime_model_from_env

    model_path = tmp_path / "sogdian.mlmodel"
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setenv("MSOCR_HTR_RUNTIME_MODEL_PATH", str(model_path))

    assert prefetch_htr_runtime_model_from_env() == model_path


def test_prefetch_htr_runtime_model_from_env_rejects_missing_path(monkeypatch, tmp_path: Path):
    from msocr.service.runtime import prefetch_htr_runtime_model_from_env

    monkeypatch.setenv("MSOCR_HTR_RUNTIME_MODEL_PATH", str(tmp_path / "missing.mlmodel"))

    with pytest.raises(FileNotFoundError, match="Configured HTR model path not found"):
        prefetch_htr_runtime_model_from_env()


def test_run_htr_service_uses_kraken_model(monkeypatch, tmp_path: Path):
    from msocr.service.runtime import run_htr_service

    image_path = tmp_path / "line.png"
    image_path.write_bytes(b"png")
    model_path = tmp_path / "sogdian.mlmodel"
    model_path.write_text("model", encoding="utf-8")

    observed = {}

    class FakeOCRModel:
        def __init__(self, mp):
            observed["model"] = str(mp)

        def set_device(self, device):
            observed["device"] = device

        def predict_line(self, image_path_arg, segmentation_type="baseline"):
            observed["image"] = str(image_path_arg)
            observed["segmentation_type"] = segmentation_type
            return {
                "image_path": str(image_path_arg),
                "predictions": [
                    {"text": "𐼷𐼹𐼻", "confidence": 0.97, "bounding_box": [10, 20, 100, 30]},
                ],
                "full_text": "𐼷𐼹𐼻",
            }

    monkeypatch.setattr("msocr.service.runtime.OCRModel", FakeOCRModel)

    result = run_htr_service(
        lang="old_sogdian",
        image_path=image_path,
        model=str(model_path),
        variant="standard",
        device="cpu",
    )

    assert observed == {
        "image": str(image_path),
        "model": str(model_path),
        "device": "cpu",
        "segmentation_type": "baseline",
    }
    assert result["text"] == "𐼷𐼹𐼻"
    assert result["engine"] == "kraken"
    assert result["language"] == "sogdian"
    assert result["writing_mode"] == "handwritten"
    assert result["lines"] == [
        {"text": "𐼷𐼹𐼻", "confidence": 0.97, "bounding_box": [10, 20, 100, 30]},
    ]
