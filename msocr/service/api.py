"""FastAPI backend for msocr."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from msocr.service.runtime import run_htr_service, run_printed_service

app = FastAPI(
    title="msocr API",
    version="0.1.0",
    description="Backend API for manuscript OCR/HTR routes.",
)


class OCRRequest(BaseModel):
    lang: str = Field(..., description="Target OCR language")
    image_path: str = Field(..., description="Path to input image")
    model: Optional[str] = Field(
        default=None, description="Optional model path override"
    )
    engine: Literal["auto", "kraken", "tesseract"] = "auto"
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "msocr-api"}


@app.post("/ocr")
def ocr(request: OCRRequest) -> dict:
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
