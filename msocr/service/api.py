"""FastAPI backend for msocr."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from msocr.service.runtime import (
    prefetch_printed_runtime_model_from_env,
    run_htr_service,
    run_printed_service,
)

app = FastAPI(
    title="msocr API",
    version="0.1.0",
    description="Backend API for manuscript OCR/HTR routes.",
)


def _sanitize_validation_payload(value: Any) -> Any:
    """Convert non-JSON-safe validation payload values into safe representations."""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {k: _sanitize_validation_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_validation_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_validation_payload(v) for v in value)
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Some invalid multipart requests include raw bytes in validation details;
    # sanitize them to avoid UTF-8 decode failures in JSON encoding.
    detail = _sanitize_validation_payload(exc.errors())
    return JSONResponse(status_code=422, content={"detail": detail})


class OCRRequest(BaseModel):
    lang: str = Field(..., description="Target OCR language")
    image_path: str = Field(..., description="Path to input image")
    model: Optional[str] = Field(
        default=None, description="Optional model path override"
    )
    engine: Literal["auto", "kraken", "tesseract", "ocrmypdf"] = "auto"
    variant: Literal["default", "estrangela", "serto", "east"] = "default"
    reference_text_path: Optional[str] = None
    cer_threshold: float = 0.05
    device: str = "cpu"


class HTRRequest(BaseModel):
    lang: str = Field(..., description="Target HTR language")
    image_path: str = Field(..., description="Path to input image")
    model: Optional[str] = Field(
        default=None, description="Optional model path override"
    )
    provider: Literal["auto", "kraken", "transkribus"] = "auto"
    device: str = "cpu"


def _run_ocr(request: OCRRequest) -> dict:
    image = Path(request.image_path)
    if not image.exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {image}")

    if request.model and not Path(request.model).exists():
        raise HTTPException(status_code=400, detail=f"Model not found: {request.model}")

    try:
        result = run_printed_service(
            lang=request.lang,
            image_path=image,
            model=request.model,
            engine=request.engine,
            variant=request.variant,
            reference_text_path=request.reference_text_path,
            cer_threshold=request.cer_threshold,
            device=request.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, **result}


def _extract_upload_file(form: Any) -> UploadFile | None:
    for key in ("file", "image", "image_file", "upload"):
        candidate = form.get(key)
        if isinstance(candidate, (UploadFile, StarletteUploadFile)):
            return candidate

    for _, value in form.multi_items():
        if isinstance(value, (UploadFile, StarletteUploadFile)):
            return value

    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "msocr-api"}


@app.on_event("startup")
def prefetch_runtime_models() -> None:
    # Production deployments can point the printed route at a HAR-backed artifact.
    prefetch_printed_runtime_model_from_env()


@app.post("/ocr")
async def ocr(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").lower()
    temp_image_path: Path | None = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = _extract_upload_file(form)
            if upload is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Missing uploaded file in multipart payload. "
                        "Provide a file field such as 'file' or 'image'."
                    ),
                )

            suffix = Path(upload.filename or "").suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await upload.read())
                temp_image_path = Path(tmp.name)

            raw_payload: dict[str, Any] = {
                "image_path": str(temp_image_path),
                "lang": form.get("lang"),
                "model": form.get("model"),
                "engine": form.get("engine"),
                "variant": form.get("variant"),
                "reference_text_path": form.get("reference_text_path"),
                "cer_threshold": form.get("cer_threshold"),
                "device": form.get("device"),
            }
            payload = {k: v for k, v in raw_payload.items() if v not in (None, "")}
            ocr_request = OCRRequest.model_validate(payload)
        else:
            body = await request.json()
            ocr_request = OCRRequest.model_validate(body)

        return _run_ocr(ocr_request)
    except ValidationError as exc:
        detail = _sanitize_validation_payload(exc.errors())
        raise HTTPException(status_code=422, detail=detail) from exc
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)


@app.post("/htr")
def htr(request: HTRRequest) -> dict:
    image = Path(request.image_path)
    if not image.exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {image}")

    if request.model and not Path(request.model).exists():
        raise HTTPException(status_code=400, detail=f"Model not found: {request.model}")

    try:
        result = run_htr_service(
            lang=request.lang,
            image_path=image,
            model=request.model,
            provider=request.provider,
            device=request.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, **result}
