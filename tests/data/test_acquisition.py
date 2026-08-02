"""Tests for reproducible external manuscript-image acquisition."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from msocr.data.acquisition import acquire_source_images, load_source_manifest


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _source_manifest(tmp_path: Path, image_bytes: bytes) -> Path:
    manifest = {
        "manifest_id": "test-source-v1",
        "base_dir": str(tmp_path / "images"),
        "source": {"base_url": "https://example.test/images/"},
        "allowed_hosts": ["example.test"],
        "image_fields": [
            "image_set",
            "filename",
            "side",
            "bytes",
            "width",
            "height",
            "sha256",
        ],
        "images": [
            [
                "I01",
                "sample.jpg",
                "recto",
                len(image_bytes),
                12,
                8,
                hashlib.sha256(image_bytes).hexdigest(),
            ]
        ],
        "unit_fields": [
            "e27_id",
            "shelfmarks",
            "image_set",
            "status",
            "catalogue_page",
            "note",
        ],
        "units": [["E27/1a", "n1", "I01", "surviving", 1, None]],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_acquire_source_images_downloads_then_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_bytes = _jpeg_bytes()
    manifest_path = _source_manifest(tmp_path, image_bytes)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response(image_bytes)

    monkeypatch.setattr("msocr.data.acquisition.urlopen", fake_urlopen)

    first = acquire_source_images(manifest_path)
    second = acquire_source_images(manifest_path, verify_only=True)

    assert first["downloaded"] == 1
    assert first["verified"] == 1
    assert second["downloaded"] == 0
    assert second["verified"] == 1
    assert calls == [("https://example.test/images/sample.jpg", 30.0)]


def test_acquire_source_images_rejects_changed_local_copy(tmp_path: Path):
    image_bytes = _jpeg_bytes()
    manifest_path = _source_manifest(tmp_path, image_bytes)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"changed")

    with pytest.raises(ValueError, match="Size mismatch"):
        acquire_source_images(manifest_path, verify_only=True)


def test_e27_source_manifest_covers_confirmed_inventory():
    manifest = load_source_manifest("data/manifests/c2-e27-dta-sources-v1.json")

    assert manifest.manifest_id == "c2-e27-dta-sources-v1"
    assert len(manifest.images) == 120
    assert len(manifest.units) == 113
    assert sum(unit["status"] == "lost" for unit in manifest.units) == 10
