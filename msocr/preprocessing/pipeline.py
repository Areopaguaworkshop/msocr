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
    isolate_mounted_fragment,
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
    line_proposal_count: int = 0


def run_fragment_pipeline(
    page_image_path: Path,
    output_dir: Path,
    *,
    expected_lines: int | None = None,
    sauvola_window: int = 51,
    min_component_area: int = 50,
    min_fragment_area: int = 5000,
    dbscan_eps: int = 150,
    isolation_mode: str = "components",
    propose_lines: bool = False,
    segmentation_model: str | None = None,
) -> PipelineResult:
    """Chain isolate → deskew → binarize → manuscript_area → row_bands.

    Outputs to ``<output_dir>/<plate_stem>_fragments/``.
    """
    page_image_path = Path(page_image_path)
    out = output_dir / f"{page_image_path.stem}_fragments"
    out.mkdir(parents=True, exist_ok=True)
    fragments_dir = out / "fragments"
    masks_dir = out / "geometry_masks"
    fragments_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    page = Image.open(page_image_path).convert("RGB")

    # 1. isolate fragments (CC + DBSCAN)
    physical_masks: dict[str, np.ndarray] = {}
    if isolation_mode == "mounted":
        mounted = isolate_mounted_fragment(page)
        if mounted is None:
            raise ValueError(
                "Could not isolate the mounted manuscript fragment; "
                "use manual ROI/region annotation"
            )
        fragment, physical_mask = mounted
        fragments = [fragment]
        physical_masks[fragment.fragment_id] = physical_mask
        Image.fromarray(physical_mask).save(out / "isolation_mask.png")
    elif isolation_mode == "components":
        fragments = isolate_fragments(
            page,
            sauvola_window=sauvola_window,
            min_component_area=min_component_area,
            min_fragment_area=min_fragment_area,
            dbscan_eps=dbscan_eps,
        )
    else:
        raise ValueError("isolation_mode must be 'mounted' or 'components'")
    fragments_to_json(fragments, out / "fragments.json")
    write_fragment_overlay(page, fragments, out / "fragments_overlay.jpg")

    # 2. per-fragment deskew + binarize for geometry; also build whole-page mask
    page_mask = np.full((page.height, page.width), 255, dtype=np.uint8)
    for frag in fragments:
        left, top, right, bottom = frag.bbox
        crop = page.crop(frag.bbox)
        physical_mask = physical_masks.get(frag.fragment_id)
        if physical_mask is not None:
            local_mask = physical_mask[top:bottom, left:right]
            crop_array = np.array(crop)
            crop_array[local_mask == 0] = 255
            crop = Image.fromarray(crop_array)
        crop_mask = binarize_for_geometry(crop, window_size=sauvola_window)
        deskewed, angle = deskew_fragment(crop, crop_mask)
        # Geometry must be recomputed after rotation; the pre-deskew mask is
        # not spatially aligned with the image Kraken/annotators will see.
        deskewed_mask = binarize_for_geometry(deskewed, window_size=sauvola_window)
        deskewed.save(fragments_dir / f"{frag.fragment_id}.png")
        Image.fromarray(deskewed_mask).save(masks_dir / f"{frag.fragment_id}_mask.png")
        (fragments_dir / f"{frag.fragment_id}_meta.json").write_text(
            json.dumps(
                {
                    "fragment_id": frag.fragment_id,
                    "source_image": str(page_image_path),
                    "source_bbox": list(frag.bbox),
                    "isolation_mode": isolation_mode,
                    "deskew_angle": angle,
                    "coordinate_space": "deskewed-fragment-local",
                    "requires_overlay_review": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # stamp fragment mask into the whole-page geometry mask
        page_mask[top:bottom, left:right] = np.minimum(
            page_mask[top:bottom, left:right], crop_mask
        )

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
            page_image_path,
            out,
            expected_lines=expected_lines,
            roi=roi,
            min_component_area=min_component_area,
        )

    proposal_count = 0
    if propose_lines:
        from msocr.segmentation.kraken_blla import segment_pages
        from msocr.segmentation.line_extraction import extract_lines_from_segments

        segments_dir = out / "line_proposals"
        segment_pages(
            fragments_dir,
            segments_dir,
            {"reading_order": "rtl", "model": segmentation_model or "blla"},
        )
        proposal_count = extract_lines_from_segments(
            fragments_dir,
            segments_dir,
            out / "line_crops",
            provenance_path=out / "line_proposal_provenance.json",
            contact_sheet_path=out / "line_proposal_contact_sheet.jpg",
        )

    return PipelineResult(
        fragments=fragments,
        manuscript_roi=roi,
        geometry_mask_path=mask_path,
        row_bands=bands,
        line_proposal_count=proposal_count,
        output_dir=out,
    )


if __name__ == "__main__":
    # Self-check: synthetic page with 3 dark rectangles (mirrors fragment_isolation self-check).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        page_path = td / "plate.png"
        page = Image.new("RGB", (800, 600), (255, 255, 255))
        for x0, y0, x1, y1 in [
            (80, 80, 220, 160),
            (380, 260, 540, 360),
            (120, 440, 280, 520),
        ]:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    page.putpixel((x, y), (20, 20, 20))
        page.save(page_path)

        res = run_fragment_pipeline(page_path, td, expected_lines=None)

        assert (
            len(res.fragments) == 3
        ), f"expected 3 fragments, got {len(res.fragments)}"
        assert (res.output_dir / "fragments.json").exists()
        assert (res.output_dir / "fragments_overlay.jpg").exists()
        assert (res.output_dir / "geometry_mask.png").exists()
        assert (res.output_dir / "manuscript_area.json").exists()
        for frag in res.fragments:
            assert (
                res.output_dir / "fragments" / f"{frag.fragment_id}.png"
            ).exists(), frag
            assert (
                res.output_dir / "geometry_masks" / f"{frag.fragment_id}_mask.png"
            ).exists(), frag
        print("ok")
