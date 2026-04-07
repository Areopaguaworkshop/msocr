"""Annotation API session manager for ground truth collection.

This module provides session persistence for the annotation API.
Sessions store under: msocr/data/sessions/{id}/

Session structure:
  {id}/session.json    - metadata: language, script, segmentation engine, line list
  {id}/page.tif       - normalised page image
  {id}/crops/         - line crop images
  {id}/annotations.json - transcript per line_id, updated on each save
"""

import json
import uuid
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging

from msocr.language_registry import LANGUAGE_REGISTRY, normalize_language_code

logger = logging.getLogger(__name__)


class IngestionPath(Enum):
    """Three ingestion paths for session creation."""
    BROWSER_UPLOAD = "browser_upload"
    LOCAL_FILE = "local_file"
    IIIF_MANIFEST = "iiif_manifest"


class ExportFormat(Enum):
    """Export format for annotations."""
    ALTO = "alto"
    PAGE = "page"
    TSV = "tsv"


class SegmentationEngine(Enum):
    """Segmentation engine fallback chain."""
    BLLA = "blla"          # Kraken baseline segmenter (primary)
    PAGESEG = "pageseg"    # Legacy bounding-box segmenter (fallback)
    MANUAL = "manual"       # Manual correction (always available)

@dataclass
class LineSegment:
    """Represents a text line with coordinates and transcription."""
    line_id: str
    order: int
    baseline_points: Optional[List[Tuple[int, int]]] = None
    boundary_points: Optional[List[Tuple[int, int]]] = None
    image_crop_path: Optional[str] = None
    transcript: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "line_id": self.line_id,
            "order": self.order,
            "baseline_points": self.baseline_points,
            "boundary_points": self.boundary_points,
            "image_crop_path": self.image_crop_path,
            "transcript": self.transcript,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LineSegment":
        """Create from dictionary."""
        return cls(
            line_id=data["line_id"],
            order=data["order"],
            baseline_points=data.get("baseline_points"),
            boundary_points=data.get("boundary_points"),
            image_crop_path=data.get("image_crop_path"),
            transcript=data.get("transcript"),
        )


@dataclass
class AnnotationSession:
    """Represents an annotation session."""
    session_id: str
    language: str
    script_variant: str
    segmentation_engine: SegmentationEngine
    lines: List[LineSegment] = field(default_factory=list)
    annotations: Dict[str, Dict] = field(default_factory=dict)
    ingestion_path: IngestionPath = IngestionPath.LOCAL_FILE
    source: str = ""
    needs_manual_review: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_rtl(self) -> bool:
        """Check if language requires RTL direction."""
        lang_info = LANGUAGE_REGISTRY.get(self.language, {})
        return lang_info.get("direction") == "rtl"

    def get_web_font_spec(self) -> str:
        """Get web font specification for this language."""
        lang_info = LANGUAGE_REGISTRY.get(self.language, {})
        return lang_info.get("web_font", "system-ui")

    def get_language_iso(self) -> str:
        """Get ISO 639 code for this language."""
        lang_info = LANGUAGE_REGISTRY.get(self.language, {})
        return lang_info.get("iso", self.language)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "script_variant": self.script_variant,
            "segmentation_engine": self.segmentation_engine.value,
            "lines": [line.to_dict() for line in self.lines],
            "annotations": self.annotations,
            "ingestion_path": self.ingestion_path.value,
            "source": self.source,
            "needs_manual_review": self.needs_manual_review,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AnnotationSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            language=data["language"],
            script_variant=data["script_variant"],
            segmentation_engine=SegmentationEngine(data["segmentation_engine"]),
            lines=[LineSegment.from_dict(l) for l in data.get("lines", [])],
            annotations=data.get("annotations", {}),
            ingestion_path=IngestionPath(data.get("ingestion_path", "local_file")),
            source=data.get("source", ""),
            needs_manual_review=data.get("needs_manual_review", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class SessionManager:
    """Manages annotation sessions with persistence."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.sessions_dir / session_id

    def _get_session_json_path(self, session_id: str) -> Path:
        """Get the session.json path."""
        return self._get_session_dir(session_id) / "session.json"

    def _get_annotations_json_path(self, session_id: str) -> Path:
        """Get the annotations.json path."""
        return self._get_session_dir(session_id) / "annotations.json"

    def _get_page_image_path(self, session_id: str) -> Path:
        """Get the page.tif path."""
        return self._get_session_dir(session_id) / "page.tif"

    def create_session(
        self,
        language: str,
        script_variant: str,
        ingestion_path: IngestionPath,
        source: str,
        lines: Optional[List[LineSegment]] = None,
        segmentation_engine: SegmentationEngine = SegmentationEngine.BLLA,
    ) -> AnnotationSession:
        """Create a new annotation session.
        
        Args:
            language: Language code (e.g., 'syriac', 'greek')
            script_variant: Script variant (e.g., 'estrangela', 'polytonic')
            ingestion_path: Source of the image (browser upload, local file, or IIIF)
            source: Source path or URL
            lines: Pre-segmented lines (optional)
            segmentation_engine: Segmentation engine used (default: BLLA)
        
        Returns:
            Created AnnotationSession
        """
        language = normalize_language_code(language)
        session_id = str(uuid.uuid4())[:8]
        
        # Set manual review flag for fallback segmentation
        needs_manual_review = segmentation_engine in (
            SegmentationEngine.PAGESEG,
            SegmentationEngine.MANUAL,
        )
        
        session = AnnotationSession(
            session_id=session_id,
            language=language,
            script_variant=script_variant,
            segmentation_engine=segmentation_engine,
            lines=lines or [],
            ingestion_path=ingestion_path,
            source=source,
            needs_manual_review=needs_manual_review,
        )

        # Create session directory
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "crops").mkdir(exist_ok=True)

        # Save session metadata
        self._save_session(session)

        logger.info(f"Created session {session_id} for {language}/{script_variant}")
        return session

    def get_session(self, session_id: str) -> Optional[AnnotationSession]:
        """Load an existing session by ID.
        
        Args:
            session_id: The session ID to load
        
        Returns:
            AnnotationSession if found, None otherwise
        """
        session_json_path = self._get_session_json_path(session_id)
        
        if not session_json_path.exists():
            logger.warning(f"Session {session_id} not found")
            return None

        try:
            with open(session_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = AnnotationSession.from_dict(data)
            
            # Load annotations if they exist
            annotations_path = self._get_annotations_json_path(session_id)
            if annotations_path.exists():
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                    session.annotations = annotations
            
            return session

        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def save_annotations(
        self, session_id: str, annotations: Dict[str, Dict]
    ) -> Optional[AnnotationSession]:
        """Save annotations to a session.
        
        Args:
            session_id: The session ID
            annotations: Dict mapping line_id to {transcript, skip}
        
        Returns:
            Updated AnnotationSession or None if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        # Update annotations
        session.annotations = annotations
        session.updated_at = datetime.now().isoformat()

        # Save annotations to file
        annotations_path = self._get_annotations_json_path(session_id)
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        # Update session.json
        self._save_session(session)

        logger.info(f"Saved {len(annotations)} annotations for session {session_id}")
        return session

    def export_session(
        self, session_id: str, format: ExportFormat
    ) -> Optional[str]:
        """Export session to specified format.
        
        Args:
            session_id: The session ID
            format: Export format (ALTO, PAGE, or TSV)
        
        Returns:
            Exported content as string, or None if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        if format == ExportFormat.ALTO:
            return self._export_alto(session)
        elif format == ExportFormat.PAGE:
            return self._export_page_xml(session)
        elif format == ExportFormat.TSV:
            return self._export_tsv(session)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _save_session(self, session: AnnotationSession):
        """Save session metadata to session.json."""
        session_json_path = self._get_session_json_path(session.session_id)
        
        with open(session_json_path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def _export_alto(self, session: AnnotationSession) -> str:
        """Export session to ALTO XML format."""
        # Create ALTO XML structure
        alto = ET.Element("alto", xmlns="http://www.loc.gov/standards/alto/ns-v4#")
        
        # Description
        description = ET.SubElement(alto, "Description")
        measurement_unit = ET.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"
        
        # Tags for script variant
        tags = ET.SubElement(alto, "Tags")
        other_tag = ET.SubElement(tags, "OtherTag")
        other_tag.set("ID", f"script_{session.script_variant}")
        other_tag.set("LABEL", session.script_variant)
        
        # Source image info
        source_image_info = ET.SubElement(description, "sourceImageInformation")
        file_name = ET.SubElement(source_image_info, "fileName")
        file_name.text = session.source
        
        # Layout
        layout = ET.SubElement(alto, "Layout")
        page = ET.SubElement(layout, "Page")
        page.set("ID", "page_1")
        page.set("PHYSICAL_IMG_NR", "1")
        
        print_space = ET.SubElement(page, "PrintSpace")
        
        # Process lines
        for line in session.lines:
            line_id = line.line_id
            annotation = session.annotations.get(line_id, {})
            
            # Skip if marked for skip
            if annotation.get("skip", False):
                continue
            
            transcript = annotation.get("transcript", "")
            
            # Get coordinates
            if line.boundary_points:
                min_x = min(p[0] for p in line.boundary_points)
                max_x = max(p[0] for p in line.boundary_points)
                min_y = min(p[1] for p in line.boundary_points)
                max_y = max(p[1] for p in line.boundary_points)
                width = max_x - min_x
                height = max_y - min_y
            else:
                min_x, min_y, width, height = 0, 0, 100, 20
            
            # Create TextBlock with LANG and TAGREFS
            textblock = ET.SubElement(print_space, "TextBlock")
            textblock.set("ID", f"block_{line.order}")
            textblock.set("LANG", session.language)
            textblock.set("TAGREFS", f"script_{session.script_variant}")
            textblock.set("HPOS", str(min_x))
            textblock.set("VPOS", str(min_y))
            textblock.set("WIDTH", str(width))
            textblock.set("HEIGHT", str(height))
            
            # Create TextLine
            textline = ET.SubElement(textblock, "TextLine")
            textline.set("ID", line_id)
            textline.set("HPOS", str(min_x))
            textline.set("VPOS", str(min_y))
            textline.set("WIDTH", str(width))
            textline.set("HEIGHT", str(height))
            
            # Add baseline if available
            if line.baseline_points:
                baseline_str = " ".join(f"{x},{y}" for x, y in line.baseline_points)
                baseline = ET.SubElement(textline, "Baseline")
                baseline.set("POINTS", baseline_str)
            
            # Add String with transcript
            if transcript:
                string_elem = ET.SubElement(textline, "String")
                string_elem.set("ID", f"string_{line.order}")
                string_elem.set("CONTENT", transcript)
        
        # Convert to string
        tree = ET.ElementTree(alto)
        
        # Use explicit encoding
        import io
        output = io.BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue().decode("utf-8")

    def _export_page_xml(self, session: AnnotationSession) -> str:
        """Export session to PAGE XML format."""
        # Create root
        root = ET.Element("PcGts", xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")
        
        # Metadata
        metadata = ET.SubElement(root, "Metadata")
        
        # Page
        page = ET.SubElement(root, "Page")
        page.set("imageFilename", session.source)
        
        # Process lines
        for line in session.lines:
            line_id = line.line_id
            annotation = session.annotations.get(line_id, {})
            
            if annotation.get("skip", False):
                continue
            
            transcript = annotation.get("transcript", "")
            
            # Create TextRegion with custom attribute
            textregion = ET.SubElement(page, "TextRegion")
            textregion.set("id", f"region_{line.order}")
            
            # Custom attribute with language and script
            custom = f"language:{{ value:{session.language}; }} script:{{ value:{session.script_variant}; }}"
            textregion.set("custom", custom)
            
            # TextLine
            textline = ET.SubElement(textregion, "TextLine")
            textline.set("id", line_id)
            
            # Coords
            if line.boundary_points:
                coords = ET.SubElement(textline, "Coords")
                points_str = " ".join(f"{x},{y}" for x, y in line.boundary_points)
                coords.set("points", points_str)
            
            # Baseline
            if line.baseline_points:
                baseline = ET.SubElement(textline, "Baseline")
                points_str = " ".join(f"{x},{y}" for x, y in line.baseline_points)
                baseline.set("points", points_str)
            
            # TextEquiv
            if transcript:
                textequiv = ET.SubElement(textline, "TextEquiv")
                unicode = ET.SubElement(textequiv, "Unicode")
                unicode.text = transcript
        
        # Convert to string
        tree = ET.ElementTree(root)
        
        import io
        output = io.BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue().decode("utf-8")

    def _export_tsv(self, session: AnnotationSession) -> str:
        """Export session to TSV format for ketos train.
        
        Format: image_path<tab>transcript
        """
        lines_output = []
        
        for line in session.lines:
            line_id = line.line_id
            annotation = session.annotations.get(line_id, {})
            
            if annotation.get("skip", False):
                continue
            
            transcript = annotation.get("transcript", "")
            image_path = line.image_crop_path or f"crops/{line_id}.jpg"
            
            # Format: image_path<tab>transcript
            lines_output.append(f"{image_path}\t{transcript}")
        
        return "\n".join(lines_output)

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        session_dirs = [d.name for d in self.sessions_dir.iterdir() if d.is_dir()]
        return sorted(session_dirs)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its files.
        
        Args:
            session_id: The session ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        session_dir = self._get_session_dir(session_id)
        
        if not session_dir.exists():
            return False
        
        shutil.rmtree(session_dir)
        logger.info(f"Deleted session {session_id}")
        return True

    def save_page_image(
        self, 
        session_id: str, 
        image_source: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
    ) -> Optional[Path]:
        """Save page image to session directory.
        
        Supports three ingestion paths:
        - LOCAL_FILE: image_source is a local file path to copy from
        - BROWSER_UPLOAD: image_bytes contains the uploaded image data
        - IIIF_MANIFEST: image_url would be fetched and downloaded (requires requests)
        
        Args:
            session_id: The session ID
            image_source: Path to source image file (for LOCAL_FILE ingestion)
            image_bytes: Raw image bytes (for BROWSER_UPLOAD ingestion)
        
        Returns:
            Path to saved page.tif, or None if failed
        """
        page_path = self._get_page_image_path(session_id)
        
        # Handle browser upload (image_bytes provided)
        if image_bytes is not None:
            page_path.write_bytes(image_bytes)
            logger.info(f"Saved page image from browser upload to {page_path}")
            return page_path
        
        # Handle local file (copy from source)
        if image_source is not None:
            if not image_source.exists():
                logger.error(f"Source image not found: {image_source}")
                return None
            shutil.copy2(image_source, page_path)
            logger.info(f"Copied page image from {image_source} to {page_path}")
            return page_path
        
        logger.warning(f"No image source provided for session {session_id}")
        return None
