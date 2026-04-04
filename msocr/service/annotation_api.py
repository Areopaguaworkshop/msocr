"""Annotation API FastAPI service for ground truth collection.

This module provides a browser-accessible API for annotation sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from msocr.data.session_manager import (
    ExportFormat,
    IngestionPath,
    LANGUAGE_REGISTRY,
    SessionManager,
)


# Pydantic request models
class CreateSessionRequest(BaseModel):
    """Request model for creating a new annotation session."""

    language: str = Field(..., description="Language code (e.g., 'syriac', 'greek')")
    script_variant: str = Field(..., description="Script variant (e.g., 'estrangela', 'polytonic')")
    ingestion_path: str = Field(
        default="browser_upload",
        description="Source of the image: browser_upload, local_file, or iiif_manifest",
    )
    source: str = Field(..., description="Source path or URL")


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
    direction: str
    web_font: str
    annotations: Dict[str, Dict]
    source: str
    lines: List[Dict]
    created_at: str
    updated_at: str


def create_app(base_dir: Optional[Path] = None) -> FastAPI:
    """Create FastAPI application for annotation API.

    Args:
        base_dir: Base directory for session storage. Defaults to temp directory.

    Returns:
        FastAPI application instance
    """
    if base_dir is None:
        import tempfile

        base_dir = Path(tempfile.mkdtemp())

    sessions_dir = base_dir / "sessions"
    manager = SessionManager(sessions_dir=sessions_dir)

    app = FastAPI(
        title="msocr Annotation API",
        version="0.1.0",
        description="API for ground truth collection and annotation",
    )

    @app.post("/api/sessions", response_model=Dict[str, Any])
    def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
        """Create a new annotation session.

        Args:
            request: Session creation request

        Returns:
            Session metadata including session_id, language, direction, and web_font

        Raises:
            HTTPException: If language is not supported
        """
        # Validate language
        if request.language not in LANGUAGE_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}. Supported: {list(LANGUAGE_REGISTRY.keys())}",
            )

        # Parse ingestion path
        try:
            ingestion = IngestionPath(request.ingestion_path)
        except ValueError:
            ingestion = IngestionPath.BROWSER_UPLOAD

        # Create session via manager
        session = manager.create_session(
            language=request.language,
            script_variant=request.script_variant,
            ingestion_path=ingestion,
            source=request.source,
        )

        # Get language metadata
        lang_info = LANGUAGE_REGISTRY[request.language]

        return {
            "session_id": session.session_id,
            "language": session.language,
            "script_variant": session.script_variant,
            "direction": lang_info["direction"],
            "web_font": lang_info["web_font"],
            "source": session.source,
            "created_at": session.created_at,
        }

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

        lang_info = LANGUAGE_REGISTRY.get(session.language, {})

        return {
            "session_id": session.session_id,
            "language": session.language,
            "script_variant": session.script_variant,
            "direction": lang_info.get("direction", "ltr"),
            "web_font": lang_info.get("web_font", "system-ui"),
            "annotations": session.annotations,
            "source": session.source,
            "lines": [line.to_dict() for line in session.lines],
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    @app.post("/api/sessions/{session_id}/save")
    def save_annotations(session_id: str, request: SaveAnnotationsRequest) -> Dict[str, str]:
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

        return {"status": "ok", "message": f"Saved {len(annotations)} annotations"}

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

        if format.lower() == "tsv":
            return Response(content=content, media_type="text/plain; charset=utf-8")
        else:
            return Response(content=content, media_type="application/xml; charset=utf-8")

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

        # Return the image
        from fastapi.responses import FileResponse

        return FileResponse(crop_path, media_type="image/jpeg")

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

    return app
