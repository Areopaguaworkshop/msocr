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
import mimetypes
import tempfile
import uuid
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import logging

from PIL import Image

from msocr.language_registry import LANGUAGE_REGISTRY, normalize_language_code
from msocr.utils.input_loader import expand_input_to_images

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
    page_count: int = 1
    needs_manual_review: bool = False
    crop_offset: Tuple[int, int] = (0, 0)
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
            "page_count": self.page_count,
            "needs_manual_review": self.needs_manual_review,
            "crop_offset": list(self.crop_offset),
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
            page_count=data.get("page_count", 1),
            needs_manual_review=data.get("needs_manual_review", False),
            crop_offset=tuple(data.get("crop_offset", (0, 0))),
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

    def _get_crops_dir(self, session_id: str) -> Path:
        """Get the crops directory path."""
        return self._get_session_dir(session_id) / "crops"

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
            language: Language code (sogdian or old_sogdian)
            script_variant: Sogdian manuscript variant label
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

    def populate_session(
        self,
        session_id: str,
        *,
        image_source: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
        iiif_manifest_url: Optional[str] = None,
        crop_manuscript_area: bool = True,
    ) -> Optional[AnnotationSession]:
        """Materialize a session's source page, segmentation, and line crops."""
        session = self.get_session(session_id)
        if session is None:
            return None

        page_path, page_count = self._materialize_session_page(
            session,
            image_source=image_source,
            image_bytes=image_bytes,
            iiif_manifest_url=iiif_manifest_url,
        )
        segmentation_engine, lines = self._segment_page_into_lines(
            session, page_path, crop_manuscript_area=crop_manuscript_area
        )

        session.page_count = page_count
        session.segmentation_engine = segmentation_engine
        session.lines = lines
        session.needs_manual_review = segmentation_engine != SegmentationEngine.BLLA
        session.updated_at = datetime.now().isoformat()
        self._save_session(session)
        logger.info(
            "Populated session %s with %s lines using %s",
            session_id,
            len(lines),
            segmentation_engine.value,
        )
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

    def _materialize_session_page(
        self,
        session: AnnotationSession,
        *,
        image_source: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
        iiif_manifest_url: Optional[str] = None,
    ) -> Tuple[Path, int]:
        page_path = self._get_page_image_path(session.session_id)
        session_dir = self._get_session_dir(session.session_id)

        with tempfile.TemporaryDirectory(prefix=f"{session.session_id}-ingest-", dir=session_dir) as td:
            temp_dir = Path(td)
            source_path = self._resolve_source_path(
                session,
                temp_dir,
                image_source=image_source,
                image_bytes=image_bytes,
                iiif_manifest_url=iiif_manifest_url,
            )
            page_images = expand_input_to_images(source_path, temp_dir / "pages")
            if not page_images:
                raise ValueError(f"No page images could be extracted from {source_path}")

            with Image.open(page_images[0]) as img:
                normalized = img.convert("RGB") if img.mode not in ("RGB", "L") else img.copy()
                normalized.save(page_path, format="TIFF")

            return page_path, len(page_images)

    def _resolve_source_path(
        self,
        session: AnnotationSession,
        temp_dir: Path,
        *,
        image_source: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
        iiif_manifest_url: Optional[str] = None,
    ) -> Path:
        if image_bytes is not None:
            suffix = Path(session.source).suffix or ".bin"
            upload_path = temp_dir / f"upload{suffix}"
            upload_path.write_bytes(image_bytes)
            return upload_path

        if image_source is not None:
            if not image_source.exists():
                raise FileNotFoundError(f"Source image not found: {image_source}")
            return image_source

        if iiif_manifest_url is not None:
            image_bytes, suffix = self._fetch_iiif_image(iiif_manifest_url)
            iiif_path = temp_dir / f"iiif{suffix}"
            iiif_path.write_bytes(image_bytes)
            return iiif_path

        raise ValueError("No source image data provided for session materialization")

    def _fetch_iiif_image(self, manifest_url: str) -> Tuple[bytes, str]:
        with urlopen(manifest_url) as response:
            manifest = json.loads(response.read().decode("utf-8"))

        image_url = self._extract_iiif_image_url(manifest)
        with urlopen(image_url) as response:
            image_bytes = response.read()
            content_type = response.headers.get_content_type()

        suffix = (
            Path(urlparse(image_url).path).suffix
            or mimetypes.guess_extension(content_type or "")
            or ".jpg"
        )
        return image_bytes, suffix

    def _extract_iiif_image_url(self, manifest: Dict) -> str:
        # IIIF Presentation API v3
        items = manifest.get("items") or []
        if items:
            canvas = items[0]
            anno_pages = canvas.get("items") or []
            if anno_pages:
                annotations = anno_pages[0].get("items") or []
                if annotations:
                    body = annotations[0].get("body") or {}
                    body_id = body.get("id") or body.get("@id")
                    if body_id:
                        return body_id

        # IIIF Presentation API v2
        sequences = manifest.get("sequences") or []
        if sequences:
            canvases = sequences[0].get("canvases") or []
            if canvases:
                images = canvases[0].get("images") or []
                if images:
                    resource = images[0].get("resource") or {}
                    resource_id = resource.get("@id") or resource.get("id")
                    if resource_id:
                        return resource_id

        raise ValueError("Could not extract an image URL from IIIF manifest")

    def _segment_page_into_lines(
        self, session: AnnotationSession, page_path: Path,
        crop_manuscript_area: bool = True,
    ) -> Tuple[SegmentationEngine, List[LineSegment]]:
        with Image.open(page_path) as img:
            working_image = img.convert("L")

            # ponytail: union bbox of all ink components above a min area, padded.
            # No density-based filtering, no header/footer heuristics — add those
            # if marginal noise is common. On any failure, pass through unchanged.
            crop_offset = (0, 0)
            if crop_manuscript_area:
                from msocr.segmentation.manuscript_area import (
                    detect_manuscript_area,
                    crop_to_manuscript_area,
                )
                try:
                    roi = detect_manuscript_area(working_image)
                    working_image, off_x, off_y = crop_to_manuscript_area(working_image, roi)
                    crop_offset = (off_x, off_y)
                except Exception as exc:
                    logger.warning(
                        "manuscript area detection failed for session %s: %s",
                        session.session_id, exc,
                    )
            width, height = working_image.size
            session.crop_offset = crop_offset

            try:
                blla_seg = self._run_blla_segmentation(working_image, session.language)
                blla_geometries = self._segmentation_to_geometries(blla_seg)
                if len(blla_geometries) >= 3:
                    return SegmentationEngine.BLLA, self._write_line_crops(
                        session.session_id, page_path, blla_geometries
                    )
            except Exception as exc:
                logger.warning("BLLA segmentation failed for session %s: %s", session.session_id, exc)

            try:
                pageseg_seg = self._run_pageseg_segmentation(working_image, session.language)
                pageseg_geometries = self._segmentation_to_geometries(pageseg_seg)
                if pageseg_geometries:
                    return SegmentationEngine.PAGESEG, self._write_line_crops(
                        session.session_id, page_path, pageseg_geometries
                    )
            except Exception as exc:
                logger.warning(
                    "pageseg segmentation failed for session %s: %s", session.session_id, exc
                )

        manual_geometry = {
            "bbox": (0, 0, max(width - 1, 0), max(height - 1, 0)),
            "boundary_points": [
                (0, 0),
                (0, max(height - 1, 0)),
                (max(width - 1, 0), max(height - 1, 0)),
                (max(width - 1, 0), 0),
            ],
            "baseline_points": None,
        }
        return SegmentationEngine.MANUAL, self._write_line_crops(
            session.session_id, page_path, [manual_geometry]
        )

    def _run_blla_segmentation(self, image: Image.Image, language: str):
        from kraken.tasks import SegmentationTaskModel
        from kraken.configs import SegmentationInferenceConfig

        seg_model = SegmentationTaskModel.load_model()
        config = SegmentationInferenceConfig(text_direction=self._text_direction(language))
        return seg_model.predict(image, config)

    def _run_pageseg_segmentation(self, image: Image.Image, language: str):
        from kraken.pageseg import segment

        return segment(image, text_direction=self._text_direction(language))

    def _text_direction(self, language: str) -> str:
        lang_info = LANGUAGE_REGISTRY.get(language, {})
        return "horizontal-rl" if lang_info.get("direction") == "rtl" else "horizontal-lr"

    def _segmentation_to_geometries(self, seg) -> List[Dict[str, object]]:
        geometries: List[Dict[str, object]] = []
        for line in getattr(seg, "lines", []):
            bbox = getattr(line, "bbox", None)
            boundary = getattr(line, "boundary", None)
            baseline = getattr(line, "baseline", None)

            boundary_points = self._normalize_points(boundary)
            baseline_points = self._normalize_points(baseline)
            if bbox and len(bbox) == 4:
                bbox_tuple = tuple(int(v) for v in bbox)
            elif boundary_points:
                xs = [point[0] for point in boundary_points]
                ys = [point[1] for point in boundary_points]
                bbox_tuple = (min(xs), min(ys), max(xs), max(ys))
            else:
                continue

            if not boundary_points:
                left, top, right, bottom = bbox_tuple
                boundary_points = [
                    (left, top),
                    (left, bottom),
                    (right, bottom),
                    (right, top),
                ]

            geometries.append(
                {
                    "bbox": bbox_tuple,
                    "boundary_points": boundary_points,
                    "baseline_points": baseline_points,
                }
            )
        return geometries

    def _normalize_points(self, points) -> Optional[List[Tuple[int, int]]]:
        if not points:
            return None
        return [(int(point[0]), int(point[1])) for point in points]

    def _write_line_crops(
        self,
        session_id: str,
        page_path: Path,
        geometries: List[Dict[str, object]],
    ) -> List[LineSegment]:
        crops_dir = self._get_crops_dir(session_id)
        for old_crop in crops_dir.glob("line_*.jpg"):
            old_crop.unlink()

        line_segments: List[LineSegment] = []
        with Image.open(page_path) as img:
            width, height = img.size
            for order, geometry in enumerate(geometries, start=1):
                bbox = self._padded_bbox(geometry["bbox"], width, height)
                crop = img.crop(bbox).convert("RGB")
                filename = f"line_{order:03d}.jpg"
                crop_path = crops_dir / filename
                crop.save(crop_path, format="JPEG")

                line_segments.append(
                    LineSegment(
                        line_id=f"line_{order:03d}",
                        order=order,
                        baseline_points=geometry.get("baseline_points"),
                        boundary_points=geometry.get("boundary_points"),
                        image_crop_path=f"crops/{filename}",
                    )
                )
        return line_segments

    def _clamp_bbox(
        self, bbox: Tuple[int, int, int, int], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = (int(v) for v in bbox)
        left = min(max(left, 0), max(width - 1, 0))
        top = min(max(top, 0), max(height - 1, 0))
        right = min(max(right, left + 1), max(width, left + 1))
        bottom = min(max(bottom, top + 1), max(height, top + 1))
        return left, top, right, bottom

    def _padded_bbox(
        self, bbox: Tuple[int, int, int, int], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = (int(v) for v in bbox)
        line_height = max(bottom - top, 1)
        pad_x = max(24, line_height // 2)
        pad_y = max(8, line_height // 5)
        return self._clamp_bbox(
            (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
            width,
            height,
        )

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

        # Map line coords back from cropped-page space to original page space.
        off_x, off_y = session.crop_offset
        
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
                min_x = min(p[0] for p in line.boundary_points) + off_x
                max_x = max(p[0] for p in line.boundary_points) + off_x
                min_y = min(p[1] for p in line.boundary_points) + off_y
                max_y = max(p[1] for p in line.boundary_points) + off_y
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
                baseline_str = " ".join(f"{x + off_x},{y + off_y}" for x, y in line.baseline_points)
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
        off_x, off_y = session.crop_offset
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
                points_str = " ".join(f"{x + off_x},{y + off_y}" for x, y in line.boundary_points)
                coords.set("points", points_str)
            
            # Baseline
            if line.baseline_points:
                baseline = ET.SubElement(textline, "Baseline")
                points_str = " ".join(f"{x + off_x},{y + off_y}" for x, y in line.baseline_points)
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
