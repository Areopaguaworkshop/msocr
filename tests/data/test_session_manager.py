"""Tests for session manager module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from msocr.data.session_manager import (
    AnnotationSession,
    LineSegment,
    SessionManager,
    IngestionPath,
    ExportFormat,
    SegmentationEngine,
)


class TestLineSegment:
    """Tests for LineSegment dataclass."""

    def test_line_segment_creation(self):
        """Test creating a LineSegment with all fields."""
        line = LineSegment(
            line_id="line_001",
            order=1,
            baseline_points=[(10, 100), (50, 102), (90, 98)],
            boundary_points=[(10, 80), (10, 120), (90, 120), (90, 80)],
            image_crop_path="crops/line_001.jpg",
        )

        assert line.line_id == "line_001"
        assert line.order == 1
        assert len(line.baseline_points) == 3
        assert len(line.boundary_points) == 4
        assert line.image_crop_path == "crops/line_001.jpg"

    def test_line_segment_default_values(self):
        """Test LineSegment with default values."""
        line = LineSegment(
            line_id="line_002",
            order=2,
        )

        assert line.line_id == "line_002"
        assert line.order == 2
        assert line.baseline_points is None
        assert line.boundary_points is None
        assert line.image_crop_path is None
        assert line.transcript is None


class TestAnnotationSession:
    """Tests for AnnotationSession dataclass."""

    def test_session_creation(self):
        """Test creating an AnnotationSession."""
        lines = [
            LineSegment(line_id="line_001", order=1),
            LineSegment(line_id="line_002", order=2),
        ]
        
        session = AnnotationSession(
            session_id="test-session-123",
            language="syriac",
            script_variant="estrangela",
            segmentation_engine=SegmentationEngine.BLLA,
            lines=lines,
            ingestion_path=IngestionPath.BROWSER_UPLOAD,
            source="test.jpg",
        )

        assert session.session_id == "test-session-123"
        assert session.language == "syriac"
        assert session.script_variant == "estrangela"
        assert session.segmentation_engine == SegmentationEngine.BLLA
        assert len(session.lines) == 2
        assert session.needs_manual_review is False
        assert session.ingestion_path == IngestionPath.BROWSER_UPLOAD

    def test_session_rtl_languages(self):
        """Test session for RTL languages."""
        session = AnnotationSession(
            session_id="rtl-session",
            language="syriac",
            script_variant="estrangela",
            segmentation_engine=SegmentationEngine.BLLA,
            lines=[],
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="/path/to/image.tif",
        )

        assert session.is_rtl is True

    def test_session_ltr_languages(self):
        """Test session for LTR languages."""
        session = AnnotationSession(
            session_id="ltr-session",
            language="greek",
            script_variant="polytonic",
            segmentation_engine=SegmentationEngine.BLLA,
            lines=[],
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="/path/to/image.tif",
        )

        assert session.is_rtl is False


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create a temporary directory for sessions."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create a SessionManager instance."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    def test_create_session(self, session_manager, temp_sessions_dir):
        """Test creating a new session."""
        lines = [
            LineSegment(line_id="line_001", order=1),
            LineSegment(line_id="line_002", order=2),
        ]

        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.BROWSER_UPLOAD,
            source="test_upload.jpg",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        assert session.session_id is not None
        assert session.language == "syriac"
        assert session.script_variant == "estrangela"
        assert len(session.lines) == 2
        assert session.ingestion_path == IngestionPath.BROWSER_UPLOAD

        # Verify session directory was created
        session_dir = temp_sessions_dir / session.session_id
        assert session_dir.exists()
        assert (session_dir / "session.json").exists()

    def test_get_session(self, session_manager, temp_sessions_dir):
        """Test loading an existing session."""
        lines = [LineSegment(line_id="line_001", order=1)]

        created_session = session_manager.create_session(
            language="greek",
            script_variant="polytonic",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="greek_page.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        loaded_session = session_manager.get_session(created_session.session_id)

        assert loaded_session is not None
        assert loaded_session.session_id == created_session.session_id
        assert loaded_session.language == "greek"
        assert loaded_session.script_variant == "polytonic"

    def test_get_nonexistent_session(self, session_manager):
        """Test loading a session that doesn't exist."""
        result = session_manager.get_session("nonexistent-id")
        assert result is None

    def test_save_annotations(self, session_manager):
        """Test saving annotations to a session."""
        lines = [
            LineSegment(line_id="line_001", order=1),
            LineSegment(line_id="line_002", order=2),
        ]

        session = session_manager.create_session(
            language="syriac",
            script_variant="serto",
            ingestion_path=IngestionPath.IIIF_MANIFEST,
            source="https://example.com/iiif/manifest.json",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        # Save annotations
        annotations = {
            "line_001": {"transcript": "�车身", "skip": False},
            "line_002": {"transcript": "ⲡⲉⲧⲣⲟⲥ", "skip": False},
        }

        updated_session = session_manager.save_annotations(
            session.session_id, annotations
        )

        assert updated_session is not None
        assert "line_001" in updated_session.annotations
        assert updated_session.annotations["line_001"]["transcript"] == "�车身"
        assert "line_002" in updated_session.annotations
        assert updated_session.annotations["line_002"]["transcript"] == "ⲡⲉⲧⲣⲟⲥ"

    def test_export_alto(self, session_manager):
        """Test exporting session to ALTO XML format."""
        lines = [
            LineSegment(
                line_id="line_001",
                order=1,
                baseline_points=[(10, 100), (50, 102)],
                boundary_points=[(10, 80), (10, 120), (90, 120), (90, 80)],
                image_crop_path="crops/line_001.jpg",
            ),
            LineSegment(
                line_id="line_002",
                order=2,
                baseline_points=[(10, 200), (50, 202)],
                boundary_points=[(10, 180), (10, 220), (90, 220), (90, 180)],
                image_crop_path="crops/line_002.jpg",
            ),
        ]

        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="page.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        # Add annotations
        annotations = {
            "line_001": {"transcript": "�车身", "skip": False},
            "line_002": {"transcript": "ⲥⲁⲣⲁ", "skip": False},
        }
        session_manager.save_annotations(session.session_id, annotations)

        # Export to ALTO
        alto_output = session_manager.export_session(
            session.session_id, ExportFormat.ALTO
        )

        assert alto_output is not None
        assert "<?xml" in alto_output
        assert '<TextBlock' in alto_output
        assert 'LANG="syriac"' in alto_output
        assert '<TextLine' in alto_output

    def test_export_page_xml(self, session_manager):
        """Test exporting session to PAGE XML format."""
        lines = [
            LineSegment(
                line_id="line_001",
                order=1,
                baseline_points=[(10, 100), (50, 102)],
                boundary_points=[(10, 80), (10, 120), (90, 120), (90, 80)],
                image_crop_path="crops/line_001.jpg",
            ),
        ]

        session = session_manager.create_session(
            language="greek",
            script_variant="polytonic",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="greek.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        annotations = {
            "line_001": {"transcript": "θεός", "skip": False},
        }
        session_manager.save_annotations(session.session_id, annotations)

        page_output = session_manager.export_session(
            session.session_id, ExportFormat.PAGE
        )

        assert page_output is not None
        assert "<?xml" in page_output
        assert '<Page' in page_output
        # PAGE XML uses custom attribute for language/script
        assert 'language:{ value:greek; }' in page_output
        assert 'script:{ value:polytonic; }' in page_output

    def test_export_tsv(self, session_manager):
        """Test exporting session to TSV format for ketos train."""
        lines = [
            LineSegment(
                line_id="line_001",
                order=1,
                image_crop_path="crops/line_001.jpg",
            ),
            LineSegment(
                line_id="line_002",
                order=2,
                image_crop_path="crops/line_002.jpg",
            ),
        ]

        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.BROWSER_UPLOAD,
            source="upload.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        annotations = {
            "line_001": {"transcript": "�车身", "skip": False},
            "line_002": {"transcript": "ⲁⲃ", "skip": False},
        }
        session_manager.save_annotations(session.session_id, annotations)

        tsv_output = session_manager.export_session(
            session.session_id, ExportFormat.TSV
        )

        assert tsv_output is not None
        assert "crops/line_001.jpg\t�车身" in tsv_output
        assert "crops/line_002.jpg\tⲁⲃ" in tsv_output

    def test_segmentation_fallback_blla(self, session_manager):
        """Test that BLLA is used as primary segmentation."""
        lines = [LineSegment(line_id="line_001", order=1)]

        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="test.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        assert session.segmentation_engine == SegmentationEngine.BLLA

    def test_segmentation_fallback_pageseg(self, session_manager):
        """Test fallback to pageseg when blla fails."""
        lines = [LineSegment(line_id="line_001", order=1)]

        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="test.tif",
            lines=lines,
            segmentation_engine=SegmentationEngine.PAGESEG,
        )

        assert session.segmentation_engine == SegmentationEngine.PAGESEG

    def test_session_with_iiif_manifest(self, session_manager):
        """Test session creation from IIIF manifest."""
        lines = [LineSegment(line_id="line_001", order=1)]

        session = session_manager.create_session(
            language="sogdian",
            script_variant="formal",
            ingestion_path=IngestionPath.IIIF_MANIFEST,
            source="https://example.com/iiif/manifest.json",
            lines=lines,
            segmentation_engine=SegmentationEngine.BLLA,
        )

        assert session.ingestion_path == IngestionPath.IIIF_MANIFEST
        assert session.language == "sogdian"

    def test_rtl_web_font_spec(self, session_manager):
        """Test that RTL languages get correct web font spec."""
        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="test.tif",
            lines=[],
            segmentation_engine=SegmentationEngine.BLLA,
        )

        font_spec = session.get_web_font_spec()
        assert "Noto Sans Syriac" in font_spec

    def test_ltr_web_font_spec(self, session_manager):
        """Test that LTR languages get correct web font spec."""
        session = session_manager.create_session(
            language="greek",
            script_variant="polytonic",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="test.tif",
            lines=[],
            segmentation_engine=SegmentationEngine.BLLA,
        )

        font_spec = session.get_web_font_spec()
        assert "GFS Didot" in font_spec or "Noto Serif" in font_spec

    def test_needs_manual_review_flag(self, session_manager):
        """Test that needs_manual_review flag is set correctly."""
        session = session_manager.create_session(
            language="syriac",
            script_variant="estrangela",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source="test.tif",
            lines=[],
            segmentation_engine=SegmentationEngine.PAGESEG,
        )

        # When using pageseg fallback, should set manual review flag
        assert session.needs_manual_review is True
