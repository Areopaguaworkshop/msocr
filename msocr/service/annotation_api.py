"""Annotation API FastAPI service for ground truth collection.

This module provides a browser-accessible API for annotation sessions, plus
serves the React SPA (``frontend/dist/``) for the annotation UI.
"""

from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile


from msocr.data.session_manager import (
    ExportFormat,
    IngestionPath,
    LANGUAGE_REGISTRY,
    SessionManager,
)
from msocr.language_registry import is_supported_language, normalize_language_code

logger = logging.getLogger(__name__)

# ponytail: SPA bundle lives outside the package (designer owns frontend/).
# Relative to cwd at app startup — mirrors the api.py / demo conventions.
_FRONTEND_DIST = Path("frontend/dist")
_FRONTEND_INDEX = _FRONTEND_DIST / "index.html"


def _spa_index() -> HTMLResponse:
    """Serve the SPA shell, or 503 with build instructions if dist is missing."""
    if not _FRONTEND_INDEX.exists():
        return HTMLResponse(
            "<h1>Annotation UI not built</h1>"
            "<p>Build the frontend:</p>"
            "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>",
            status_code=503,
        )
    return FileResponse(_FRONTEND_INDEX, media_type="text/html")


# Pydantic request models
class CreateSessionRequest(BaseModel):
    """Request model for creating a new annotation session."""

    language: str = Field(..., description="Language code: sogdian or old_sogdian")
    script_variant: str = Field(..., description="Sogdian manuscript variant label")
    ingestion_path: str = Field(
        default="browser_upload",
        description="Source of the image: browser_upload, local_file, or iiif_manifest",
    )
    source: Optional[str] = Field(default=None, description="Source path or URL")
    crop_manuscript_area: bool = Field(
        default=True,
        description="Detect and crop to manuscript area before line segmentation",
    )


class AnnotationItem(BaseModel):
    """Single annotation item."""

    line_id: str = Field(..., description="Line identifier")
    transcript: Optional[str] = Field(default="", description="Transcribed text")
    skip: bool = Field(default=False, description="Whether to skip this line")


class SaveAnnotationsRequest(BaseModel):
    """Request model for saving annotations."""

    annotations: List[Dict[str, Any]] = Field(
        ..., description="List of annotations with line_id and transcript"
    )


class SessionResponse(BaseModel):
    """Response model for session data."""

    session_id: str
    language: str
    script_variant: str
    segmentation_engine: str
    page_count: int
    line_count: int
    direction: str
    web_font: str
    needs_manual_review: bool
    annotations: Dict[str, Dict]
    source: str
    lines: List[Dict]
    created_at: str
    updated_at: str


def _extract_upload_file(form: Any) -> UploadFile | None:
    for key in ("file", "image", "image_file", "upload"):
        candidate = form.get(key)
        if isinstance(candidate, (UploadFile, StarletteUploadFile)):
            return candidate

    for _, value in form.multi_items():
        if isinstance(value, (UploadFile, StarletteUploadFile)):
            return value

    return None


def _serialize_session(session) -> Dict[str, Any]:
    lang_info = LANGUAGE_REGISTRY.get(session.language, {})
    return {
        "session_id": session.session_id,
        "language": session.language,
        "script_variant": session.script_variant,
        "segmentation_engine": session.segmentation_engine.value,
        "page_count": session.page_count,
        "line_count": len(session.lines),
        "direction": lang_info.get("direction", "ltr"),
        "web_font": lang_info.get("web_font", "system-ui"),
        "needs_manual_review": session.needs_manual_review,
        "annotations": session.annotations,
        "annotations_v2": session.annotations_v2,
        "source": session.source,
        "lines": [line.to_dict() for line in session.lines],
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def create_app(base_dir: Optional[Path] = None, crop_manuscript_area: bool = True) -> FastAPI:
    """Create FastAPI application for annotation API.

    Args:
        base_dir: Base directory for session storage. Defaults to temp directory.
        crop_manuscript_area: App-level default for manuscript-area cropping.
            A request's ``crop_manuscript_area`` field is ANDed with this — the CLI
            ``--no-crop-manuscript-area`` flag forces it off regardless of request.

    Returns:
        FastAPI application instance
    """
    if base_dir is None:
        import tempfile

        base_dir = Path(tempfile.mkdtemp())

    sessions_dir = base_dir / "sessions"
    manager = SessionManager(sessions_dir=sessions_dir)
    app_crop_enabled = crop_manuscript_area

    app = FastAPI(
        title="msocr Annotation API",
        version="0.1.0",
        description="API for ground truth collection and annotation",
    )

    # ponytail: CORS for Vite dev server only; broaden if other frontends appear
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/sessions", response_model=List[Dict[str, Any]])
    def list_sessions() -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id, "language": s.language,
                "script_variant": s.script_variant, "source": s.source,
                "line_count": len(s.lines), "created_at": s.created_at,
                "updated_at": s.updated_at,
                "has_annotations": bool(getattr(s, "annotations_v2", {})),
            }
            for s in manager._all_sessions()
        ]

    @app.post("/api/sessions", response_model=Dict[str, Any])
    async def create_session(request: Request) -> Dict[str, Any]:
        """Create a new annotation session.

        Args:
            request: Session creation request

        Returns:
            Session metadata including session_id, language, direction, and web_font

        Raises:
            HTTPException: If language is not supported
        """
        content_type = request.headers.get("content-type", "").lower()
        upload: UploadFile | None = None
        image_bytes: bytes | None = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = _extract_upload_file(form)
            payload = {
                "language": form.get("language"),
                "script_variant": form.get("script_variant"),
                "ingestion_path": form.get("ingestion_path", "browser_upload"),
                "source": form.get("source") or getattr(upload, "filename", None),
            }
            session_request = CreateSessionRequest.model_validate(payload)
            if upload is not None:
                image_bytes = await upload.read()
        else:
            payload = await request.json()
            session_request = CreateSessionRequest.model_validate(payload)

        # Validate language
        if not is_supported_language(session_request.language):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {session_request.language}. Supported: {list(LANGUAGE_REGISTRY.keys())}",
            )
        language = normalize_language_code(session_request.language)

        # Parse ingestion path
        try:
            ingestion = IngestionPath(session_request.ingestion_path)
        except ValueError:
            ingestion = IngestionPath.BROWSER_UPLOAD

        source = session_request.source
        if not source:
            raise HTTPException(status_code=422, detail="Missing source path or uploaded filename")

        # Create session via manager
        session = manager.create_session(
            language=language,
            script_variant=session_request.script_variant,
            ingestion_path=ingestion,
            source=source,
        )

        if ingestion == IngestionPath.BROWSER_UPLOAD:
            if image_bytes is None:
                raise HTTPException(
                    status_code=422,
                    detail="Browser upload sessions require a multipart file field such as 'file' or 'image'.",
                )
            populated = manager.populate_session(
                session.session_id,
                image_bytes=image_bytes,
                crop_manuscript_area=app_crop_enabled and session_request.crop_manuscript_area,
            )
        elif ingestion == IngestionPath.LOCAL_FILE:
            populated = manager.populate_session(
                session.session_id,
                image_source=Path(source),
                crop_manuscript_area=app_crop_enabled and session_request.crop_manuscript_area,
            )
        else:
            populated = manager.populate_session(
                session.session_id,
                iiif_manifest_url=source,
                crop_manuscript_area=app_crop_enabled and session_request.crop_manuscript_area,
            )

        if populated is None:
            raise HTTPException(status_code=500, detail="Failed to populate annotation session")

        return _serialize_session(populated)

    @app.get("/api/sessions/{session_id}", response_model=Dict[str, Any])
    def get_session(session_id: str) -> Dict[str, Any]:
        """Get session state with all annotations.

        Args:
            session_id: Session ID

        Returns:
            Session metadata and annotations

        Raises:
            HTTPException: If session not found
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        return _serialize_session(session)

    @app.post("/api/sessions/{session_id}/save")
    def save_annotations(session_id: str, request: SaveAnnotationsRequest) -> Dict[str, Any]:
        """Save annotations to a session.

        Args:
            session_id: Session ID
            request: Save annotations request

        Returns:
            Success message

        Raises:
            HTTPException: If session not found
        """
        # Convert list format to dict format
        annotations = {}
        for item in request.annotations:
            line_id = item.get("line_id")
            if line_id:
                annotations[line_id] = {
                    "transcript": item.get("transcript", ""),
                    "skip": item.get("skip", False),
                }

        session = manager.save_annotations(session_id, annotations)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        return {
            "status": "ok",
            "message": f"Saved {len(annotations)} annotations",
            **_serialize_session(session),
        }

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str, format: str = "alto") -> Response:
        """Export session in specified format.

        Args:
            session_id: Session ID
            format: Export format (alto, page, or tsv)

        Returns:
            XML or text response with exported content

        Raises:
            HTTPException: If session not found or format invalid
        """
        try:
            export_format = ExportFormat(format.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {format}. Valid: alto, page, tsv",
            )

        content = manager.export_session(session_id, export_format)
        if content is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # ponytail: also persist the export into the session dir so it lives on
        # disk in a known place for Kraken training, not just in the browser
        # Downloads folder. Best-effort — a write failure shouldn't break the
        # download.
        ext = "tsv" if format.lower() == "tsv" else "xml"
        fname = f"{session_id}.{ext}"
        try:
            out_path = manager._get_session_dir(session_id) / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist %s export for %s: %s", ext, session_id, exc)

        media = "text/plain; charset=utf-8" if format.lower() == "tsv" else "application/xml; charset=utf-8"
        resp = Response(content=content, media_type=media)
        resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return resp

    @app.get("/api/sessions/{session_id}/line/{line_number}/image")
    def get_line_image(session_id: str, line_number: int) -> Response:
        """Get cropped line image.

        Args:
            session_id: Session ID
            line_number: Line number (1-indexed)

        Returns:
            Image file

        Raises:
            HTTPException: If session not found
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # Find the line by order (1-indexed)
        lines = session.lines
        if line_number < 1 or line_number > len(lines):
            raise HTTPException(
                status_code=404,
                detail=f"Line {line_number} not found (session has {len(lines)} lines)",
            )

        line = lines[line_number - 1]
        if line.image_crop_path is None:
            raise HTTPException(status_code=404, detail=f"No crop image for line {line_number}")

        # Load the crop image
        crop_path = manager._get_session_dir(session_id) / line.image_crop_path
        if not crop_path.exists():
            raise HTTPException(status_code=404, detail=f"Crop image not found: {crop_path}")

        return FileResponse(crop_path, media_type="image/jpeg")

    @app.get("/api/sessions/{session_id}/image")
    def get_session_image(session_id: str) -> Response:
        """Serve the source fragment image for the drawing UI.

        For LOCAL_FILE sessions, serve the source path directly. For
        BROWSER_UPLOAD sessions, serve the materialized page.tif.
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # ponytail: prefer the original source file for LOCAL_FILE so the UI
        # gets the deskewed fragment the user pointed at; fall back to the
        # materialized page.tif for uploads/IIIF.
        candidate: Optional[Path] = None
        if session.ingestion_path == IngestionPath.LOCAL_FILE and session.source:
            src = Path(session.source)
            if src.exists():
                candidate = src
        if candidate is None:
            page_path = manager._get_page_image_path(session_id)
            if page_path.exists():
                candidate = page_path
        if candidate is None:
            raise HTTPException(status_code=404, detail="No source image available for session")

        media_type, _ = mimetypes.guess_type(str(candidate))
        return FileResponse(candidate, media_type=media_type or "image/png")

    @app.get("/api/sessions/{session_id}/autosuggest")
    def autosuggest(session_id: str) -> Dict[str, Any]:
        """Run default Kraken BLLA on the source fragment, return regions + lines.

        Returns ``{"regions": [...], "lines": [...]}`` with the JSON shape the
        drawing UI expects. On any BLLA failure, returns empty lists with 200
        OK so the UI can start blank.
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        image_path = manager._get_page_image_path(session_id)
        if not image_path.exists() and session.ingestion_path == IngestionPath.LOCAL_FILE and session.source:
            alt = Path(session.source)
            if alt.exists():
                image_path = alt
        if not image_path.exists():
            return {"regions": [], "lines": []}

        try:
            from msocr.segmentation.kraken_blla import _serialize_segmentation
            from kraken.tasks import SegmentationTaskModel
            from kraken.configs import SegmentationInferenceConfig
            from PIL import Image

            lang_info = LANGUAGE_REGISTRY.get(session.language, {})
            text_direction = "horizontal-rl" if lang_info.get("direction") == "rtl" else "horizontal-lr"
            seg_model = SegmentationTaskModel.load_model()
            with Image.open(image_path) as img:
                seg = seg_model.predict(img, SegmentationInferenceConfig(text_direction=text_direction))
        except Exception as exc:
            logger.warning("autosuggest BLLA failed for session %s: %s", session_id, exc)
            return {"regions": [], "lines": []}

        regions_out: List[Dict[str, Any]] = []
        for rtype, region_list in (getattr(seg, "regions", None) or {}).items():
            for idx, region in enumerate(region_list, start=1):
                boundary = getattr(region, "boundary", None) or []
                regions_out.append({
                    "id": getattr(region, "id", None) or f"r{len(regions_out) + 1}",
                    "polygon": [[int(p[0]), int(p[1])] for p in boundary],
                    "type": rtype,
                })

        lines_out: List[Dict[str, Any]] = []
        for idx, line in enumerate(getattr(seg, "lines", None) or [], start=1):
            baseline = getattr(line, "baseline", None) or []
            boundary = getattr(line, "boundary", None) or []
            lines_out.append({
                "id": getattr(line, "id", None) or f"l{idx}",
                "baseline": [[int(p[0]), int(p[1])] for p in baseline],
                "boundary": [[int(p[0]), int(p[1])] for p in boundary],
                "type": "default",
            })

        return {"regions": regions_out, "lines": lines_out}

    @app.get("/api/sessions/{session_id}/annotations")
    def get_annotations_v2(session_id: str) -> Dict[str, Any]:
        """Return saved v2 drawing-UI annotation state."""
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        state = session.annotations_v2 or {"regions": [], "lines": []}
        return {"has_annotations": bool(session.annotations_v2), **state}

    @app.post("/api/sessions/{session_id}/annotations")
    async def save_annotations_v2(session_id: str, request: Request) -> Dict[str, Any]:
        """Store v2 drawing-UI annotation state (regions + lines).

        Replaces any prior v2 state. Body shape:
        {"regions": [{"id","polygon","type"}], "lines": [{"id","baseline","type","transcript"}]}
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        body = await request.json()
        regions = body.get("regions", []) or []
        lines = body.get("lines", []) or []
        updated = manager.save_annotations_v2(session_id, regions, lines)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return {"status": "ok"}

    @app.post("/api/sessions/{session_id}/import-xml")
    async def import_xml(
        session_id: str,
        xml_path: Optional[str] = None,
        file: Optional[UploadFile] = None,
    ) -> Dict[str, Any]:
        """Import v2 annotation state from a PAGE XML file (manual safety valve).

        The auto-import in get_session covers the canonical gt/{id}_page_*.xml
        path. Use this endpoint when the XML is at a non-standard location or
        uploaded from a browser.

        Accepts EITHER:
        - multipart file upload (field name ``file``), OR
        - ``xml_path`` query/body string (kept for curl/back-compat)
        """
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        if file is not None:
            xml_bytes = await file.read()
            if not xml_bytes:
                raise HTTPException(status_code=400, detail="empty file upload")
            updated = manager.import_v2_from_xml_bytes(session_id, xml_bytes)
        elif xml_path:
            p = Path(xml_path)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"XML not found: {p}")
            updated = manager.import_v2_from_xml(session_id, p)
        else:
            raise HTTPException(status_code=400, detail="either file upload or xml_path required")
        if updated is None:
            raise HTTPException(status_code=422, detail="XML parsed but no regions/lines found")
        v2 = updated.annotations_v2 or {}
        return {
            "status": "ok",
            "regions": len(v2.get("regions", [])),
            "lines": len(v2.get("lines", [])),
        }

    @app.get("/api/languages")
    def list_languages() -> List[Dict[str, Any]]:
        """List supported languages with RTL/LTR direction and web fonts.

        Returns:
            List of language metadata
        """
        languages = []
        for lang_code, info in LANGUAGE_REGISTRY.items():
            languages.append(
                {
                    "code": lang_code,
                    "direction": info["direction"],
                    "web_font": info["web_font"],
                }
            )
        return languages

    # ponytail: corpus plate listing for the C2AV Christian Sogdian manuscript.
    # Hardcoded corpus root for now — C2AV is the only manuscript in scope.
    # If a second manuscript lands, generalize to /api/corpus/{corpus_id}/plates
    # with a registry; until then YAGNI.
    _C2AV_ROOT = Path("dataset/christian_sogdian_c2av")
    _C2AV_PLATES = _C2AV_ROOT / "plates"

    @app.get("/api/corpus/c2av/plates")
    def list_c2av_plates() -> List[Dict[str, Any]]:
        """List all 99 C2AV plates with annotation status.

        Status is derived by matching plate number to existing annotation
        sessions (session_id like c2avNN, zero-padded 2 digits). A plate is
        'annotated' if a session exists for it; transcribed line count comes
        from the session's line transcripts.
        """
        if not _C2AV_PLATES.exists():
            return []
        # index sessions by plate number for O(1) lookup
        sessions_by_plate: Dict[int, Dict[str, Any]] = {}
        for s in manager._all_sessions():
            sid = s.session_id
            # session_id convention: c2avNN where NN is 2-digit plate number
            if sid.startswith("c2av") and sid[4:].isdigit():
                plate_num = int(sid[4:])
                transcribed = sum(
                    1 for ln in s.lines if ln.transcript and ln.transcript.strip()
                )
                sessions_by_plate[plate_num] = {
                    "session_id": sid,
                    "line_count": len(s.lines),
                    "transcribed_count": transcribed,
                    "updated_at": s.updated_at,
                }
        plates = []
        for plate_path in sorted(_C2AV_PLATES.glob("p-*.png")):
            stem = plate_path.stem  # e.g. p-19
            num = int(stem.split("-", 1)[1])
            plates.append({
                "plate_number": num,
                "filename": plate_path.name,
                "path": str(plate_path),
                "thumbnail_url": f"/api/corpus/c2av/plates/{plate_path.name}/thumbnail",
                "status": sessions_by_plate.get(num, None),
            })
        return plates

    @app.get("/api/corpus/c2av/plates/{filename}/thumbnail")
    def c2av_plate_thumbnail(filename: str) -> Response:
        """Serve a plate as a JPEG thumbnail (max 200px wide) for the gallery."""
        plate_path = _C2AV_PLATES / filename
        if not plate_path.exists() or not filename.endswith(".png"):
            raise HTTPException(status_code=404, detail=f"Plate not found: {filename}")
        from PIL import Image
        import io
        with Image.open(plate_path) as img:
            thumb = img.copy()
            thumb.thumbnail((200, 280))
            buf = io.BytesIO()
            thumb.convert("RGB").save(buf, format="JPEG", quality=80)
            return Response(content=buf.getvalue(), media_type="image/jpeg")

    @app.post("/api/corpus/c2av/plates/{filename}/session")
    def create_c2av_plate_session(filename: str) -> Dict[str, Any]:
        """Create an annotation session for a specific C2AV plate.

        Uses ingestion_path=local_file with the plate as source. Session ID
        follows the c2avNN convention so plate-status lookup works. Idempotent:
        if a session already exists for the plate, returns it unchanged.
        """
        plate_path = _C2AV_PLATES / filename
        if not plate_path.exists() or not filename.endswith(".png"):
            raise HTTPException(status_code=404, detail=f"Plate not found: {filename}")
        stem = plate_path.stem
        num = int(stem.split("-", 1)[1])
        session_id = f"c2av{num:02d}"
        existing = manager.get_session(session_id)
        if existing is not None:
            return _serialize_session(existing)
        session = manager.create_session(
            language="sogdian",
            script_variant="christian-syriac-script",
            ingestion_path=IngestionPath.LOCAL_FILE,
            source=str(plate_path),
            session_id_override=session_id,
        )
        populated = manager.populate_session(session_id, image_source=plate_path)
        if populated is None:
            raise HTTPException(status_code=500, detail="Failed to populate plate session")
        return _serialize_session(populated)

    # ponytail: SPA serving. /assets mounted from dist (Vite convention);
    # catch-all registered LAST so it cannot shadow /api/* routes. The React
    # router owns client-side routes like /ui/{session_id}. If the frontend
    # bundle is missing, the catch-all returns 503 with build instructions so
    # API clients still get clean responses.
    if (_FRONTEND_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def spa_catch_all(full_path: str) -> HTMLResponse:
        # Serve a real file from the dist root if it exists (favicon, etc.).
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            media_type, _ = mimetypes.guess_type(str(candidate))
            return FileResponse(candidate, media_type=media_type or "application/octet-stream")
        return _spa_index()

    return app