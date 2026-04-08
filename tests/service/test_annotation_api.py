"""Tests for annotation API endpoints."""

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


def _multipart_payload(language: str, script_variant: str, image_path: Path):
    return {
        "language": (None, language),
        "script_variant": (None, script_variant),
        "ingestion_path": (None, "browser_upload"),
        "file": (image_path.name, image_path.read_bytes(), "image/png"),
    }


def test_create_session_endpoint(tmp_path: Path):
    """Test POST /api/sessions endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "upload.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        files=_multipart_payload("syriac", "estrangela", image_path),
    )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["language"] == "syriac"
    assert data["script_variant"] == "estrangela"
    assert data["page_count"] == 1
    assert data["line_count"] >= 1
    assert data["segmentation_engine"] in {"blla", "pageseg", "manual"}


def test_get_session_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id} endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "local.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session first
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "greek",
            "script_variant": "polytonic",
            "ingestion_path": "local_file",
            "source": str(image_path),
        },
    )
    session_id = create_response.json()["session_id"]

    # Get session
    response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["language"] == "greek"
    assert data["page_count"] == 1
    assert data["line_count"] >= 1


def test_save_annotations_endpoint(tmp_path: Path):
    """Test POST /api/sessions/{id}/save endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "save.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session
    create_response = client.post(
        "/api/sessions",
        files=_multipart_payload("sogdian", "formal", image_path),
    )
    session_id = create_response.json()["session_id"]

    get_response = client.get(f"/api/sessions/{session_id}")
    line_id = get_response.json()["lines"][0]["line_id"]

    # Save annotations
    response = client.post(
        f"/api/sessions/{session_id}/save",
        json={
            "annotations": [
                {"line_id": line_id, "transcript": "𐽾𐽿"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["annotations"][line_id]["transcript"] == "𐽾𐽿"

    # Verify annotations
    get_response = client.get(f"/api/sessions/{session_id}")
    assert len(get_response.json()["annotations"]) == 1


def test_export_alto_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=alto endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "alto.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        files=_multipart_payload("syriac", "estrangela", image_path),
    )
    session_id = create_response.json()["session_id"]
    line_id = client.get(f"/api/sessions/{session_id}").json()["lines"][0]["line_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": line_id, "transcript": "ܒܪܫܝܬ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=alto")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]


def test_export_page_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=page endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "page.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        files=_multipart_payload("coptic", "sahidic", image_path),
    )
    session_id = create_response.json()["session_id"]
    line_id = client.get(f"/api/sessions/{session_id}").json()["lines"][0]["line_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": line_id, "transcript": "ⲧⲁⲓ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=page")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]


def test_export_tsv_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=tsv endpoint."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "tsv.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        files=_multipart_payload("old_turkish", "old_uyghur", image_path),
    )
    session_id = create_response.json()["session_id"]
    line_id = client.get(f"/api/sessions/{session_id}").json()["lines"][0]["line_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": line_id, "transcript": "test"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=tsv")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_rtl_language_metadata(tmp_path: Path):
    """Test that RTL languages get proper direction metadata."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "rtl.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create RTL language session
    response = client.post(
        "/api/sessions",
        files=_multipart_payload("syriac", "estrangela", image_path),
    )

    data = response.json()
    assert data["direction"] == "rtl"
    assert data["web_font"] is not None  # Noto Sans Syriac expected


def test_legacy_armenia_alias_is_normalized(tmp_path: Path):
    """Test that the legacy CLI/API typo still resolves to the canonical language."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "armenia.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        files=_multipart_payload("armenia", "bolorgir", image_path),
    )

    assert response.status_code == 200
    assert response.json()["language"] == "armenian"


def test_create_session_supports_iiif_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test IIIF manifest ingestion resolves to a hydrated session."""
    from msocr.service.annotation_api import create_app

    image = Image.new("RGB", (80, 40), color="white")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    manifest = {
        "items": [
            {
                "items": [
                    {
                        "items": [
                            {
                                "body": {"id": "https://example.com/canvas.png"}
                            }
                        ]
                    }
                ]
            }
        ]
    }

    class _Response:
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
            return _Response(json.dumps(manifest).encode("utf-8"))
        return _Response(image_bytes, content_type="image/png")

    monkeypatch.setattr("msocr.data.session_manager.urlopen", fake_urlopen)

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        json={
            "language": "greek",
            "script_variant": "polytonic",
            "ingestion_path": "iiif_manifest",
            "source": "https://example.com/manifest.json",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["page_count"] == 1
    assert data["line_count"] >= 1


def test_line_image_endpoint_returns_crop(tmp_path: Path):
    """Test GET /api/sessions/{id}/line/{n}/image returns generated crop."""
    from msocr.service.annotation_api import create_app

    image_path = _make_test_image(tmp_path / "line.png")
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions",
        files=_multipart_payload("latin", "caroline", image_path),
    )
    session_id = create_response.json()["session_id"]

    response = client.get(f"/api/sessions/{session_id}/line/1/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
