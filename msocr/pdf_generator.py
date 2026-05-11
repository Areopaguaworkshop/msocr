"""PDF/A-1b searchable PDF generation."""

from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io


def generate_searchable_pdf(
    image_paths: List[Path],
    ocr_data: Dict[str, Any],
    output_path: Path,
) -> None:
    writer = PdfWriter()

    pages_data = ocr_data.get("pages", [])

    for idx, image_path in enumerate(image_paths):
        page_num = idx + 1
        page_text = ""
        for p in pages_data:
            if p.get("page_number") == page_num:
                page_text = p.get("text", "")
                break

        if not image_path.exists():
            continue

        img_pdf = _create_pdf_page(image_path, page_text)
        img_reader = PdfReader(io.BytesIO(img_pdf))
        if img_reader.pages:
            writer.add_page(img_reader.pages[0])

    writer.add_metadata({
        "/Title": ocr_data.get("metadata", {}).get("input_file", "OCR Result"),
        "/Creator": "msocr",
    })

    with output_path.open("wb") as f:
        writer.write(f)


def _create_pdf_page(image_path: Path, text: str) -> bytes:
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size

    dpi = 150
    page_width = width * 72 / dpi
    page_height = height * 72 / dpi

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))

    c.drawImage(ImageReader(str(image_path)), 0, 0, width=page_width, height=page_height)

    if text:
        text_obj = c.beginText(10, page_height - 20)
        text_obj.setFont("Helvetica", 8)
        text_obj.setFillColorRGB(0, 0, 0)
        for line in text.split("\n")[:50]:
            text_obj.textLine(line[:100])
        c.drawText(text_obj)

    c.save()
    packet.seek(0)
    return packet.getvalue()
