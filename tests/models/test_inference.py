from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

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


def test_predict_line_crops_manuscript_area_before_page_segmentation(monkeypatch, tmp_path: Path):
    model = OCRModel.__new__(OCRModel)
    image_path = tmp_path / "page.png"
    model.model_path = tmp_path / "model.safetensors"
    model.device = "cpu"
    observed = {}

    page = Image.new("L", (220, 180), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 80, 130, 100), fill=0)

    class FakeSegModel:
        def predict(self, im, config):
            observed["segmentation_image_size"] = im.size
            return "bounds"

    class FakeRecognitionModel:
        def predict(self, im, segmentation, config):
            observed["recognition_image_size"] = im.size
            observed["segmentation"] = segmentation
            return [SimpleNamespace(prediction="𐼷", confidence=0.9)]

    model._seg_model = FakeSegModel()
    model.model = FakeRecognitionModel()
    monkeypatch.setattr(model, "preprocess_image", lambda _path: page)

    result = model.predict_line(image_path, segmentation_type="baseline")

    assert result["full_text"] == "𐼷"
    assert observed["segmentation"] == "bounds"
    assert observed["segmentation_image_size"][0] < page.size[0]
    assert observed["segmentation_image_size"][1] < page.size[1]
    assert observed["recognition_image_size"] == observed["segmentation_image_size"]
