"""Tests for the Sogdian annotation API."""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_test_image(path: Path, *, size: tuple[int, int] = (120, 60)) -> Path:
    image = Image.new("RGB", size, color="white")
    image.save(path)
    return path


def _multipart_payload(image_path: Path, *, language: str = "sogdian"):
    return {
        "language": (None, language),
        "script_variant": (None, "standard"),
        "ingestion_path": (None, "browser_upload"),
        "file": (image_path.name, image_path.read_bytes(), "image/png"),
    }


def test_create_session_endpoint(tmp_path: Path):
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "upload.png")
    client = TestClient(create_app(base_dir=tmp_path))

    response = client.post("/api/sessions", files=_multipart_payload(image_path))

    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "sogdian"
    assert data["script_variant"] == "standard"
    assert data["direction"] == "rtl"
    assert data["web_font"] == "Noto Sans Sogdian"
    assert data["line_count"] >= 1


def test_create_session_accepts_old_sogdian_alias(tmp_path: Path):
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "alias.png")
    client = TestClient(create_app(base_dir=tmp_path))

    response = client.post(
        "/api/sessions",
        files=_multipart_payload(image_path, language="old_sogdian"),
    )

    assert response.status_code == 200
    assert response.json()["language"] == "sogdian"


def test_get_save_and_export_session(tmp_path: Path):
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "page.png")
    client = TestClient(create_app(base_dir=tmp_path))

    create_response = client.post("/api/sessions", files=_multipart_payload(image_path))
    session_id = create_response.json()["session_id"]
    line_id = client.get(f"/api/sessions/{session_id}").json()["lines"][0]["line_id"]

    save_response = client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": line_id, "transcript": "𐼷𐼹𐼻"}]},
    )
    assert save_response.status_code == 200
    assert save_response.json()["annotations"][line_id]["transcript"] == "𐼷𐼹𐼻"

    alto_response = client.get(f"/api/sessions/{session_id}/export?format=alto")
    assert alto_response.status_code == 200
    assert "application/xml" in alto_response.headers["content-type"]

    tsv_response = client.get(f"/api/sessions/{session_id}/export?format=tsv")
    assert tsv_response.status_code == 200
    assert "text/plain" in tsv_response.headers["content-type"]


def test_create_session_supports_iiif_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from msocr.service.annotation_api import create_app

    image = Image.new("RGB", (80, 40), color="white")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    manifest = {
        "items": [
            {
                "items": [
                    {"items": [{"body": {"id": "https://example.com/canvas.png"}}]}
                ]
            }
        ]
    }

    class Response:
        def __init__(self, body: bytes, content_type: str = "application/json"):
            self._body = body
            self.headers = self
            self._content_type = content_type

        def read(self):
            return self._body

        def get_content_type(self):
            return self._content_type

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(url: str):
        if url.endswith("manifest.json"):
            return Response(json.dumps(manifest).encode("utf-8"))
        return Response(image_bytes, content_type="image/png")

    monkeypatch.setattr("msocr.data.session_manager.urlopen", fake_urlopen)
    client = TestClient(create_app(base_dir=tmp_path))

    response = client.post(
        "/api/sessions",
        json={
            "language": "sogdian",
            "script_variant": "standard",
            "ingestion_path": "iiif_manifest",
            "source": "https://example.com/manifest.json",
        },
    )

    assert response.status_code == 200
    assert response.json()["line_count"] >= 1


def test_line_image_endpoint_returns_crop(tmp_path: Path):
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "line.png")
    client = TestClient(create_app(base_dir=tmp_path))

    create_response = client.post("/api/sessions", files=_multipart_payload(image_path))
    session_id = create_response.json()["session_id"]

    response = client.get(f"/api/sessions/{session_id}/line/1/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
