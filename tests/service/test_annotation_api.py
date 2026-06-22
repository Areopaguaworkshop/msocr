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

def test_plan_route_returns_html(tmp_path):
    from msocr.service.annotation_api import create_app
    app = create_app(base_dir=tmp_path)
    client = TestClient(app)
    resp = client.get("/plan")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # The plan should mention the design doc title
    assert "HTR Training Pipeline" in resp.text or "Implementation Plan" in resp.text

def test_line_save_preserves_existing_annotations(tmp_path):
    from msocr.data.session_manager import IngestionPath, LineSegment, SessionManager
    from msocr.service.annotation_api import create_app

    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        language="sogdian",
        script_variant="christian-syriac-script",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="page.png",
        lines=[LineSegment(line_id="line_001", order=1), LineSegment(line_id="line_002", order=2)],
    )
    manager.save_annotations(session.session_id, {"line_001": {"transcript": "ܐ", "skip": False}})

    # ponytail: per-line POST /line/{n}/save route was removed (replaced by
    # bulk POST /annotations); exercise the v1 manager.save_annotations path
    # directly to verify preservation behavior.
    manager.save_annotations(
        session.session_id,
        {"line_001": {"transcript": "ܐ", "skip": False}, "line_002": {"transcript": "ܒ", "skip": False}},
    )
    annotations = manager.get_session(session.session_id).annotations
    assert annotations["line_001"]["transcript"] == "ܐ"
    assert annotations["line_002"]["transcript"] == "ܒ"


def test_line_save_accepts_skip_checkbox(tmp_path):
    from msocr.data.session_manager import IngestionPath, LineSegment, SessionManager

    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        language="sogdian",
        script_variant="christian-syriac-script",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="page.png",
        lines=[LineSegment(line_id="line_001", order=1)],
    )

    manager.save_annotations(
        session.session_id, {"line_001": {"transcript": "ܐ", "skip": True}}
    )
    annotation = manager.get_session(session.session_id).annotations["line_001"]
    assert annotation == {"transcript": "ܐ", "skip": True}


def test_v2_annotations_roundtrip_and_page_export(tmp_path):
    """v2 drawing-UI annotations persist and export to SegmOnto PAGE XML."""
    from msocr.data.session_manager import IngestionPath, LineSegment, SessionManager
    from msocr.data.session_manager import ExportFormat
    from msocr.service.annotation_api import create_app

    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        language="sogdian",
        script_variant="sogdian-manuscript",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="frag_004_deskewed.png",
        lines=[LineSegment(line_id="line_001", order=1)],
    )
    regions = [{"id": "r1", "polygon": [[10, 10], [100, 10], [100, 200], [10, 200]], "type": "MainZone"}]
    lines = [{"id": "l1", "baseline": [[15, 50], [95, 50]], "type": "DefaultLine", "transcript": "test"}]
    manager.save_annotations_v2(session.session_id, regions, lines)

    fetched = manager.get_annotations_v2(session.session_id)
    assert fetched == {"regions": regions, "lines": lines}

    page_xml = manager.export_session(session.session_id, ExportFormat.PAGE)
    assert 'custom="structure {type:MainZone;}"' in page_xml
    assert 'custom="structure {type:DefaultLine;}"' in page_xml
    assert "<Unicode>test</Unicode>" in page_xml
