"""Tests for annotation API endpoints."""
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_create_session_endpoint(tmp_path: Path):
    """Test POST /api/sessions endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test_page.tif",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["language"] == "syriac"
    assert data["script_variant"] == "estrangela"


def test_get_session_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id} endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session first
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "greek",
            "script_variant": "polytonic",
            "ingestion_path": "local_file",
            "source": "input/test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    # Get session
    response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["language"] == "greek"


def test_save_annotations_endpoint(tmp_path: Path):
    """Test POST /api/sessions/{id}/save endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "sogdian",
            "script_variant": "formal",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    # Save annotations
    response = client.post(
        f"/api/sessions/{session_id}/save",
        json={
            "annotations": [
                {"line_id": "line_001", "transcript": "𐽾𐽿"},
                {"line_id": "line_002", "transcript": "test"},
            ]
        },
    )

    assert response.status_code == 200

    # Verify annotations
    get_response = client.get(f"/api/sessions/{session_id}")
    assert len(get_response.json()["annotations"]) == 2


def test_export_alto_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=alto endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "ܒܪܫܝܬ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=alto")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]


def test_export_page_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=page endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "coptic",
            "script_variant": "sahidic",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "ⲧⲁⲓ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=page")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]


def test_export_tsv_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=tsv endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "old_turkish",
            "script_variant": "old_uyghur",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "test"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=tsv")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_rtl_language_metadata(tmp_path: Path):
    """Test that RTL languages get proper direction metadata."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create RTL language session
    response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )

    data = response.json()
    assert data["direction"] == "rtl"
    assert data["web_font"] is not None  # Noto Sans Syriac expected