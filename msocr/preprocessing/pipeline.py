"""Stage 1 integration: chain the existing preprocessing/segmentation modules.

ponytail: the modules exist. This file is the wiring. ~50 lines, no new
algorithms. ManuscriptPreprocessor (whole-page) stays for legacy callers;
this is the per-fragment path per docs/fix-a-v9-fragment-pipeline.md §Stage 1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from msocr.preprocessing.binarize import binarize_for_geometry
from msocr.preprocessing.deskew import deskew_fragment
from msocr.segmentation.fragment_isolation import (
    Fragment,
    fragments_to_json,
    isolate_fragments,
    write_fragment_overlay,
)
from msocr.segmentation.manuscript_area import detect_manuscript_area
from msocr.segmentation.row_bands import LineBand, extract_row_bands


@dataclass
class PipelineResult:
    fragments: list[Fragment]
    manuscript_roi: tuple[int, int, int, int] | None
    geometry_mask_path: Path
    output_dir: Path
    row_bands: list[LineBand] = field(default_factory=list)


def run_fragment_pipeline(
    page_image_path: Path,
    output_dir: Path,
    *,
    expected_lines: int | None = None,
    sauvola_window: int = 51,
    min_component_area: int = 50,
    min_fragment_area: int = 5000,
    dbscan_eps: int = 150,
) -> PipelineResult:
    """Chain isolate → deskew → binarize → manuscript_area → row_bands.

    Outputs to ``<output_dir>/<plate_stem>_fragments/``.
    """
    page_image_path = Path(page_image_path)
    out = output_dir / f"{page_image_path.stem}_fragments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fragments").mkdir(exist_ok=True)

    page = Image.open(page_image_path).convert("RGB")

    # 1. isolate fragments (CC + DBSCAN)
    fragments = isolate_fragments(
        page,
        sauvola_window=sauvola_window,
        min_component_area=min_component_area,
        min_fragment_area=min_fragment_area,
        dbscan_eps=dbscan_eps,
    )
    fragments_to_json(fragments, out / "fragments.json")
    write_fragment_overlay(page, fragments, out / "fragments_overlay.jpg")

    # 2. per-fragment deskew + binarize for geometry; also build whole-page mask
    page_mask = np.full((page.height, page.width), 255, dtype=np.uint8)
    for frag in fragments:
        left, top, right, bottom = frag.bbox
        crop = page.crop(frag.bbox)
        crop_mask = binarize_for_geometry(crop, window_size=sauvola_window)
        deskewed, angle = deskew_fragment(crop, crop_mask)
        deskewed.save(out / "fragments" / f"{frag.fragment_id}.png")
        Image.fromarray(crop_mask).save(out / "fragments" / f"{frag.fragment_id}_mask.png")
        (out / "fragments" / f"{frag.fragment_id}_meta.json").write_text(
            json.dumps(
                {"fragment_id": frag.fragment_id, "bbox": list(frag.bbox), "deskew_angle": angle},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # stamp fragment mask into the whole-page geometry mask
        page_mask[top:bottom, left:right] = np.minimum(page_mask[top:bottom, left:right], crop_mask)

    mask_path = out / "geometry_mask.png"
    Image.fromarray(page_mask).save(mask_path)

    # 3. manuscript area detection
    roi = detect_manuscript_area(page, min_area=min_component_area, margin_pad=20)
    (out / "manuscript_area.json").write_text(
        json.dumps({"roi": list(roi) if roi else None}, ensure_ascii=False),
        encoding="utf-8",
    )

    # 4. row bands (only when expected_lines given)
    bands: list[LineBand] = []
    if expected_lines is not None:
        bands = extract_row_bands(
            page_image_path, out, expected_lines=expected_lines, roi=roi,
            min_component_area=min_component_area,
        )

    return PipelineResult(
        fragments=fragments,
        manuscript_roi=roi,
        geometry_mask_path=mask_path,
        row_bands=bands,
        output_dir=out,
    )


if __name__ == "__main__":
    # Self-check: synthetic page with 3 dark rectangles (mirrors fragment_isolation self-check).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        page_path = td / "plate.png"
        page = Image.new("RGB", (800, 600), (255, 255, 255))
        for x0, y0, x1, y1 in [(80, 80, 220, 160), (380, 260, 540, 360), (120, 440, 280, 520)]:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    page.putpixel((x, y), (20, 20, 20))
        page.save(page_path)

        res = run_fragment_pipeline(page_path, td, expected_lines=None)

        assert len(res.fragments) == 3, f"expected 3 fragments, got {len(res.fragments)}"
        assert (res.output_dir / "fragments.json").exists()
        assert (res.output_dir / "fragments_overlay.jpg").exists()
        assert (res.output_dir / "geometry_mask.png").exists()
        assert (res.output_dir / "manuscript_area.json").exists()
        for frag in res.fragments:
            assert (res.output_dir / "fragments" / f"{frag.fragment_id}.png").exists(), frag
            assert (res.output_dir / "fragments" / f"{frag.fragment_id}_mask.png").exists(), frag
        print("ok")