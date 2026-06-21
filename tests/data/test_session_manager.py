"""Tests for Sogdian annotation session persistence."""

from pathlib import Path

import pytest

from msocr.data.session_manager import (
    AnnotationSession,
    ExportFormat,
    IngestionPath,
    LineSegment,
    SegmentationEngine,
    SessionManager,
)


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(sessions_dir=tmp_path / "sessions")


def test_annotation_session_uses_sogdian_metadata():
    session = AnnotationSession(
        session_id="test-session",
        language="sogdian",
        script_variant="standard",
        segmentation_engine=SegmentationEngine.BLLA,
        lines=[LineSegment(line_id="line_001", order=1)],
        ingestion_path=IngestionPath.BROWSER_UPLOAD,
        source="page.png",
    )

    assert session.is_rtl is True
    assert session.get_language_iso() == "sog"
    assert session.get_web_font_spec() == "Noto Sans Sogdian"


def test_create_session_normalizes_old_sogdian_alias(session_manager: SessionManager):
    session = session_manager.create_session(
        language="old_sogdian",
        script_variant="standard",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="page.tif",
        lines=[LineSegment(line_id="line_001", order=1)],
    )

    loaded = session_manager.get_session(session.session_id)
    assert loaded is not None
    assert loaded.language == "sogdian"
    assert loaded.script_variant == "standard"
    assert loaded.lines[0].line_id == "line_001"


def test_save_annotations_and_export_formats(session_manager: SessionManager):
    session = session_manager.create_session(
        language="sogdian",
        script_variant="standard",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="page.tif",
        lines=[
            LineSegment(
                line_id="line_001",
                order=1,
                baseline_points=[(10, 100), (90, 100)],
                boundary_points=[(10, 80), (90, 80), (90, 120), (10, 120)],
                image_crop_path="crops/line_001.jpg",
            )
        ],
    )
    session_manager.save_annotations(
        session.session_id,
        {"line_001": {"transcript": "𐼷𐼹𐼻", "skip": False}},
    )

    alto_output = session_manager.export_session(session.session_id, ExportFormat.ALTO)
    page_output = session_manager.export_session(session.session_id, ExportFormat.PAGE)
    tsv_output = session_manager.export_session(session.session_id, ExportFormat.TSV)

    assert 'LANG="sogdian"' in alto_output
    assert 'CONTENT="𐼷𐼹𐼻"' in alto_output
    assert "language:{ value:sogdian; }" in page_output
    assert "script:{ value:standard; }" in page_output
    assert "crops/line_001.jpg\t𐼷𐼹𐼻" in tsv_output


def test_manual_segmentation_sets_review_flag(session_manager: SessionManager):
    session = session_manager.create_session(
        language="sogdian",
        script_variant="standard",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source="page.tif",
        segmentation_engine=SegmentationEngine.MANUAL,
    )

    assert session.needs_manual_review is True


def test_line_crop_bbox_is_padded_and_clamped(session_manager: SessionManager):
    assert session_manager._padded_bbox((10, 10, 50, 40), 100, 80) == (0, 2, 74, 48)


def test_save_page_image_from_local_file(session_manager: SessionManager, tmp_path: Path):
    source = tmp_path / "source.tif"
    source.write_bytes(b"image_data")
    session = session_manager.create_session(
        language="sogdian",
        script_variant="standard",
        ingestion_path=IngestionPath.LOCAL_FILE,
        source=str(source),
    )

    saved = session_manager.save_page_image(session.session_id, image_source=source)

    assert saved is not None
    assert saved.name == "page.tif"
    assert saved.read_bytes() == b"image_data"
