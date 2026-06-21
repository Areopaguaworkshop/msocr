from pathlib import Path

from msocr.models.inference import OCRModel


def test_predict_page_uses_baseline_segmentation(monkeypatch, tmp_path: Path):
    model = OCRModel.__new__(OCRModel)
    image_path = tmp_path / "page.png"

    observed = {}

    def fake_predict_line(path, segmentation_type="baseline"):
        observed["path"] = path
        observed["segmentation_type"] = segmentation_type
        return {"full_text": "text"}

    monkeypatch.setattr(model, "predict_line", fake_predict_line)

    assert model.predict_page(image_path) == {"full_text": "text"}
    assert observed == {"path": image_path, "segmentation_type": "baseline"}
