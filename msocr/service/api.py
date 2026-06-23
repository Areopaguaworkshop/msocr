"""FastAPI backend for the manuscript HTR runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from msocr.service.runtime import prefetch_htr_runtime_model_from_env, run_htr_service

app = FastAPI(
    title="msocr Sogdian HTR API",
    version="0.21.0",
    description="Local Kraken HTR backend for manuscript OCR, starting with Sogdian.",
)

# ponytail: CORS for Vite dev server only; broaden if other frontends appear
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    detail = _sanitize_validation_payload(exc.errors())
    return JSONResponse(status_code=422, content={"detail": detail})


class HTRRequest(BaseModel):
    lang: str = Field(default="sogdian", description="Target manuscript language")
    image_path: str = Field(..., description="Path to input image")
    model: Optional[str] = Field(default=None, description="Optional Kraken .mlmodel path")
    variant: str = Field(default="standard", description="Sogdian manuscript variant label")
    device: str = "cpu"


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
    return {"status": "ok", "service": "msocr-htr-api", "route": "htr"}


@app.on_event("startup")
def prefetch_runtime_model() -> None:
    prefetch_htr_runtime_model_from_env()


@app.post("/htr")
async def htr(request: Request) -> dict:
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
                "variant": form.get("variant"),
                "device": form.get("device"),
            }
            payload = {k: v for k, v in raw_payload.items() if v not in (None, "")}
            htr_request = HTRRequest.model_validate(payload)
        else:
            body = await request.json()
            htr_request = HTRRequest.model_validate(body)

        image = Path(htr_request.image_path)
        if not image.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {image}")

        if htr_request.model and not Path(htr_request.model).exists():
            raise HTTPException(status_code=400, detail=f"Model not found: {htr_request.model}")

        result = run_htr_service(
            lang=htr_request.lang,
            image_path=image,
            model=htr_request.model,
            variant=htr_request.variant,
            device=htr_request.device,
        )
    except ValidationError as exc:
        detail = _sanitize_validation_payload(exc.errors())
        raise HTTPException(status_code=422, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)

    return {"ok": True, **result}
