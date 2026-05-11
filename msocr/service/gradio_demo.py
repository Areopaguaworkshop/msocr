"""Gradio browser demo for msocr."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from msocr.language_registry import DEMO_LANGUAGE_CODES
from msocr.service.runtime import (
    prefetch_htr_runtime_model_from_env,
    prefetch_printed_runtime_model_from_env,
    run_htr_service,
    run_printed_service,
)

LANGUAGES = list(DEMO_LANGUAGE_CODES)


def _save_temp_image(image: Image.Image) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        return Path(tmp.name)


def _demo_ocr(
    image: Optional[Image.Image],
    lang: str,
    engine: str,
    variant: str,
    model: str,
    reference_text_path: str,
    cer_threshold: float,
    device: str,
) -> tuple[str, str]:
    if image is None:
        return "", "error: image is required"
    image_path = _save_temp_image(image)
    model_arg = model.strip() or None
    ref_arg = reference_text_path.strip() or None
    try:
        result = run_printed_service(
            lang=lang,
            image_path=image_path,
            model=model_arg,
            engine=engine,
            variant=variant,
            reference_text_path=ref_arg,
            cer_threshold=cer_threshold,
            device=device,
        )
        return result[
            "text"
        ], f"engine={result['engine']} mode=ocr lang={result['language']}"
        return result[
            "text"
        ], f"engine={result['engine']} mode=ocr lang={result['language']}"
    except Exception as exc:
        return "", f"error: {exc}"


def _demo_htr(
    image: Optional[Image.Image],
    lang: str,
    provider: str,
    variant: str,
    model: str,
    device: str,
) -> tuple[str, str]:
    if image is None:
        return "", "error: image is required"
    image_path = _save_temp_image(image)
    model_arg = model.strip() or None
    try:
        result = run_htr_service(
            lang=lang,
            image_path=image_path,
            model=model_arg,
            provider=provider,
            variant=variant,
            device=device,
        )
        return result[
            "text"
        ], f"engine={result['engine']} mode=htr lang={result['language']}"
    except Exception as exc:
        return "", f"error: {exc}"


def build_demo() -> gr.Blocks:
    prefetch_printed_runtime_model_from_env()
    prefetch_htr_runtime_model_from_env()
    with gr.Blocks(title="msocr Demo") as demo:
        gr.Markdown("# msocr Demo\nPrinted OCR and Handwritten HTR quick test UI.")

        with gr.Tab("Printed OCR"):
            with gr.Row():
                with gr.Column():
                    ocr_image = gr.Image(type="pil", label="Input Image")
                    ocr_lang = gr.Dropdown(LANGUAGES, value="latin", label="Language")
                    ocr_engine = gr.Dropdown(
                        ["auto", "kraken", "tesseract"], value="auto", label="Engine"
                    )
                    ocr_variant = gr.Dropdown(
                        ["default", "estrangela", "serto", "east"],
                        value="default",
                        label="Variant",
                    )
                    ocr_model = gr.Textbox(
                        label="Model Path (optional)", placeholder="models/kraken/..."
                    )
                    ocr_reference = gr.Textbox(
                        label="Reference Text Path (optional)",
                        placeholder="path/to/reference.txt",
                    )
                    ocr_cer = gr.Number(value=0.05, label="CER Threshold")
                    ocr_device = gr.Textbox(value="cpu", label="Device")
                    ocr_btn = gr.Button("Run OCR")
                with gr.Column():
                    ocr_text = gr.Textbox(label="OCR Text", lines=12)
                    ocr_info = gr.Textbox(label="Run Info")

            ocr_btn.click(
                _demo_ocr,
                inputs=[
                    ocr_image,
                    ocr_lang,
                    ocr_engine,
                    ocr_variant,
                    ocr_model,
                    ocr_reference,
                    ocr_cer,
                    ocr_device,
                ],
                outputs=[ocr_text, ocr_info],
            )

        with gr.Tab("Handwritten HTR"):
            with gr.Row():
                with gr.Column():
                    htr_image = gr.Image(type="pil", label="Input Image")
                    htr_lang = gr.Dropdown(LANGUAGES, value="latin", label="Language")
                    htr_provider = gr.Dropdown(
                        ["auto", "kraken", "transkribus"],
                        value="auto",
                        label="Provider",
                    )
                    htr_variant = gr.Textbox(value="default", label="Variant")
                    htr_model = gr.Textbox(
                        label="Model Path (optional)", placeholder="models/kraken/..."
                    )
                    htr_device = gr.Textbox(value="cpu", label="Device")
                    htr_btn = gr.Button("Run HTR")
                with gr.Column():
                    htr_text = gr.Textbox(label="HTR Text", lines=12)
                    htr_info = gr.Textbox(label="Run Info")

            htr_btn.click(
                _demo_htr,
                inputs=[htr_image, htr_lang, htr_provider, htr_variant, htr_model, htr_device],
                outputs=[htr_text, htr_info],
            )

    return demo
