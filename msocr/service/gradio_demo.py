"""Gradio browser demo for Sogdian manuscript HTR."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from msocr.service.runtime import prefetch_htr_runtime_model_from_env, run_htr_service


def _save_temp_image(image: Image.Image) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        return Path(tmp.name)


def _demo_htr(
    image: Optional[Image.Image],
    variant: str,
    model: str,
    device: str,
) -> tuple[str, str]:
    if image is None:
        return "", "error: image is required"
    image_path = _save_temp_image(image)
    model_arg = model.strip() or None
    variant_arg = variant.strip() or "standard"
    try:
        result = run_htr_service(
            lang="sogdian",
            image_path=image_path,
            model=model_arg,
            variant=variant_arg,
            device=device,
        )
        return result["text"], (
            f"engine={result['engine']} mode=htr lang={result['language']} "
            f"variant={result['variant']}"
        )
    except Exception as exc:
        return "", f"error: {exc}"
    finally:
        image_path.unlink(missing_ok=True)


def build_demo() -> gr.Blocks:
    prefetch_htr_runtime_model_from_env()
    with gr.Blocks(title="msocr Sogdian HTR") as demo:
        gr.Markdown(
            "# msocr Sogdian HTR\n"
            "Local Kraken manuscript OCR/HTR for Sogdian manuscript images."
        )

        with gr.Row():
            with gr.Column():
                htr_image = gr.Image(type="pil", label="Sogdian Manuscript Image")
                htr_variant = gr.Textbox(value="standard", label="Variant")
                htr_model = gr.Textbox(
                    label="Kraken Model Path (optional)",
                    placeholder="models/kraken/sogdian_manuscript.mlmodel",
                )
                htr_device = gr.Textbox(value="cpu", label="Device")
                htr_btn = gr.Button("Run Sogdian HTR")
            with gr.Column():
                htr_text = gr.Textbox(label="HTR Text", lines=14)
                htr_info = gr.Textbox(label="Run Info")

        htr_btn.click(
            _demo_htr,
            inputs=[htr_image, htr_variant, htr_model, htr_device],
            outputs=[htr_text, htr_info],
        )

    return demo
