"""Input loaders for image/PDF HTR entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import List


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def expand_input_to_images(
    input_path: Path, temp_dir: Path, dpi: int = 300
) -> List[Path]:
    """Expand image/PDF input into image paths.

    - Image input returns a single-item list with the original image path.
    - PDF input renders all pages to PNGs in `temp_dir` and returns page image paths.
    
    PDF input is rendered to temporary page images before Kraken inference.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if not is_pdf(input_path):
        return [input_path]

    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "PDF input requires pypdfium2. Install dependency and retry."
        ) from exc

    doc = pdfium.PdfDocument(str(input_path))
    if len(doc) == 0:
        raise ValueError(f"PDF has no pages: {input_path}")

    out: List[Path] = []
    scale = dpi / 72.0
    for i in range(len(doc)):
        page = doc[i]
        pil_image = page.render(scale=scale).to_pil()
        page_path = temp_dir / f"{input_path.stem}_page_{i + 1:04d}.png"
        pil_image.save(page_path)
        out.append(page_path)
        page.close()
    doc.close()
    return out
