#!/usr/bin/env python3
"""Phase 0.5 lacuna augmentation: blank a random subset of lines per plate.

Mirrors ``scripts/augment_lines.py`` but instead of degrading every line crop,
it fully blanks a random ``--blank-fraction`` subset and rewrites those lines'
``<TextEquiv><Unicode>`` to ``LACUNA_TOKEN`` (░) in the copied XML. The
recognizer thus sees blank line images paired with the lacuna token, learning
to emit ░ (with low confidence) for blank/damaged input.

Lines NOT chosen keep their original crop + transcript. Output plates reuse
the original PAGE XML structure (only imageFilename + chosen lines' Unicode
change) so the existing page-format training pipeline consumes them without
changes.

ponytail: same synthetic-plate reassembly as augment_lines.py. The only delta
is "blank some lines + rewrite their transcripts". Upgrade path: partial-blank
(blank only part of the crop width) when full blank proves too easy.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import lxml.etree as ET
import numpy as np
from PIL import Image

from kraken.lib.xml import parse_page
from kraken.lib.segmentation import calculate_polygonal_environment

from msocr.training.lacuna_augment import LACUNA_TOKEN, lacuna_blank_line

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _line_bboxes(polys: list) -> list[tuple[int, int, int, int] | None]:
    """Return (x0, y0, x1, y1) bbox per polygon, None where poly is None.

    ponytail: copied verbatim from scripts/augment_lines.py — that file is a
    script, not an importable module, so refactoring it into a shared helper
    would be more disruption than the ~12 lines are worth.
    """
    boxes = []
    for p in polys:
        if p is None:
            boxes.append(None)
            continue
        arr = np.asarray(p)
        x0, y0 = int(arr[:, 0].min()), int(arr[:, 1].min())
        x1, y1 = int(arr[:, 0].max()), int(arr[:, 1].max())
        boxes.append((x0, y0, x1, y1))
    return boxes


def augment_lacuna_plate(
    img_path: Path,
    xml_path: Path,
    out_img_path: Path,
    out_xml_path: Path,
    variant_seed: int,
    blank_fraction: float,
) -> tuple[int, int]:
    """Produce one lacuna-augmented plate image + copied XML.

    Returns (line_count, blanked_count).
    """
    rng = np.random.RandomState(variant_seed)
    img = Image.open(str(img_path)).convert("L")
    doc = ET.parse(str(xml_path))
    res = parse_page(doc, xml_path, "baselines")
    lines = list(res["lines"].values())
    baselines = [l.baseline for l in lines]
    polys = calculate_polygonal_environment(
        im=img, baselines=baselines, topline=False, raise_on_error=False
    )
    boxes = _line_bboxes(polys)

    n_lines = len(lines)
    n_blank = int(round(n_lines * blank_fraction))
    blank_idx = set(rng.choice(n_lines, size=max(0, n_blank), replace=False).tolist()) if n_blank else set()

    canvas = img.copy()
    for idx, (line, poly, box) in enumerate(zip(lines, polys, boxes)):
        if idx not in blank_idx or poly is None or box is None:
            continue
        x0, y0, x1, y1 = box
        crop = img.crop((x0, y0, x1, y1))
        blank = lacuna_blank_line(crop)
        # ponytail: paste at bbox top-left, crop to bbox size so we don't
        # overwrite neighbors (blank is same size as crop, so this is exact).
        canvas.paste(blank.crop((0, 0, min(blank.size[0], x1 - x0), min(blank.size[1], y1 - y0))), (x0, y0))

    canvas.save(str(out_img_path))

    # Copy XML, rewrite imageFilename to ABSOLUTE path (matches original c2av
    # convention — see dataset/christian_sogdian_c2av/gt/*.xml which uses
    # absolute imageFilename paths) + blanked lines' Unicode.
    shutil.copyfile(str(xml_path), str(out_xml_path))
    tree = ET.parse(str(out_xml_path))
    page = tree.getroot().find(f"{{{NS}}}Page")
    if page is not None:
        page.set("imageFilename", str(out_img_path.resolve()))
    # ponytail: walk TextLines in document order, match by index to blank_idx.
    text_lines = tree.getroot().findall(f".//{{{NS}}}TextLine")
    for idx, tl in enumerate(text_lines):
        if idx not in blank_idx:
            continue
        te = tl.find(f"{{{NS}}}TextEquiv")
        if te is None:
            te = ET.SubElement(tl, f"{{{NS}}}TextEquiv")
        uni = te.find(f"{{{NS}}}Unicode")
        if uni is None:
            uni = ET.SubElement(te, f"{{{NS}}}Unicode")
        uni.text = LACUNA_TOKEN
    tree.write(str(out_xml_path), xml_declaration=True, encoding="utf-8")
    return n_lines, len(blank_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0.5 lacuna augmentation (blank a random subset of lines).")
    ap.add_argument("--manifest", default="data/manifests/c2av-finetune.json")
    ap.add_argument("--out-dir", default="dataset/christian_sogdian_c2av/lacuna")
    ap.add_argument("--out-manifest", default="data/manifests/c2av-finetune-lacuna.json")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--blank-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    base_dir = Path(manifest["base_dir"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = out_dir / "plates"
    aug_xml_dir = out_dir / "gt"
    aug_dir.mkdir(exist_ok=True)
    aug_xml_dir.mkdir(exist_ok=True)

    train_entries = manifest["partitions"]["train"]
    val_entries = manifest["partitions"]["validation"]
    holdout_entries = manifest["partitions"]["holdout"]

    new_train = []
    total_lines = 0
    total_blanked = 0
    for entry in train_entries:
        new_train.append(entry)  # keep original
        xml_rel = entry["xml_path"]
        xml_path = base_dir / xml_rel
        tree = ET.parse(str(xml_path))
        page = tree.getroot().find(f"{{{NS}}}Page")
        img_name = page.get("imageFilename")
        img_path = xml_path.parent / img_name
        ms_id = entry["manuscript_id"]
        for v in range(args.variants):
            seed = args.seed + hash((ms_id, v)) % (2**31)
            out_img = aug_dir / f"{ms_id}_lacuna_v{v}.png"
            out_xml = aug_xml_dir / f"{ms_id}_lacuna_v{v}.xml"
            n, nb = augment_lacuna_plate(
                img_path, xml_path, out_img, out_xml, seed, args.blank_fraction
            )
            total_lines += n
            total_blanked += nb
            new_train.append(
                {
                    "id": f"{ms_id}-lacuna-v{v}",
                    "manuscript_id": ms_id,
                    "xml_path": str((aug_xml_dir / out_xml.name).relative_to(out_dir.parent)),
                    "augmented": True,
                    "augment_kind": "lacuna",
                }
            )

    aug_manifest = dict(manifest)
    aug_manifest["manifest_id"] = manifest["manifest_id"] + "-lacuna"
    aug_manifest["base_dir"] = str(out_dir.parent)
    aug_manifest["partitions"] = {
        "train": new_train,
        "validation": val_entries,
        "holdout": holdout_entries,
    }
    aug_manifest["metadata"] = dict(manifest.get("metadata", {}))
    aug_manifest["metadata"]["lacuna_variants"] = args.variants
    aug_manifest["metadata"]["lacuna_blank_fraction"] = args.blank_fraction
    aug_manifest["metadata"]["lacuna_train_plates"] = len(new_train)
    aug_manifest["metadata"]["lacuna_total_lines"] = total_lines + manifest["metadata"].get("train_lines", 0)
    aug_manifest["metadata"]["lacuna_blanked_lines"] = total_blanked

    Path(args.out_manifest).write_text(json.dumps(aug_manifest, indent=2, ensure_ascii=False))
    print(
        f"Phase 0.5 lacuna augment done: {len(new_train)} train plates "
        f"({len(train_entries)} orig + {len(new_train) - len(train_entries)} lacuna), "
        f"{total_blanked}/{total_lines} lines blanked. "
        f"Manifest → {args.out_manifest}"
    )


if __name__ == "__main__":
    main()