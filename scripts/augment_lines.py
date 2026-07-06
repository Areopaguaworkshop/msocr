#!/usr/bin/env python3
"""§3 line-level augmentation for the 148-line Sogdian fine-tune set.

Applies heavier degradation than ketos `--augment` (which uses albumentations
at 50% activation) by running the ported ocropy/Kanungo pipeline
(distort_line → degrade_line → ocropy_degrade) on every line crop, 3 variants
per plate with distinct seeds. Output plates reuse the original PAGE XML
annotations (only imageFilename changes) so the existing page-format training
pipeline consumes them without changes.

Ponytail: synthetic-plate reassembly keeps the page-format pipeline unchanged.
A line-level training path would need a separate ketos format + manifest; not
worth it for 148 lines. Upgrade path: if line-level training is needed later,
write the augmented crops as .png+.gt.txt pairs and switch the orchestrator
to `-f path`.
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

from msocr.training.line_augment import distort_line, degrade_line, ocropy_degrade

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
VARIANTS = 3
# ponytail: heavier than DefaultAugmenter (albumentations 50% activation, no
# Kanungo noise). lib-2 reference: distort(distort=4, sigma=8, eps=0.05,
# delta=0.5) then degrade(eta=0.02, alpha=1.2, beta=1.2). We chain ocropy_degrade
# on top for additional ocropus-style noise. Ceiling: if this still overfits,
# add affine-only variants (no geometric distortion) for shape diversity.
DISTORT_KW = dict(distort=4.0, sigma=8.0, eps=0.05, delta=0.5)
DEGRADE_KW = dict(eta=0.02, alpha=1.2, beta=1.2)
OCROPY_KW = dict(distort=0.5, dsigma=10.0, eps=0.02, delta=0.2)


def _line_bboxes(polys: list) -> list[tuple[int, int, int, int] | None]:
    """Return (x0, y0, x1, y1) bbox per polygon, None where poly is None."""
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


def augment_plate(
    img_path: Path,
    xml_path: Path,
    out_img_path: Path,
    out_xml_path: Path,
    out_image_basename: str,
    variant_seed: int,
) -> int:
    """Produce one augmented plate image + copied XML. Returns line count."""
    rng = np.random.RandomState(variant_seed)
    np.random.seed(variant_seed)  # line_augment uses np.random.* directly
    img = Image.open(str(img_path)).convert("L")
    doc = ET.parse(str(xml_path))
    res = parse_page(doc, xml_path, "baselines")
    lines = list(res["lines"].values())
    baselines = [l.baseline for l in lines]
    polys = calculate_polygonal_environment(im=img, baselines=baselines, topline=False, raise_on_error=False)
    boxes = _line_bboxes(polys)
    canvas = img.copy()
    n = 0
    for line, poly, box in zip(lines, polys, boxes):
        if poly is None or box is None:
            continue
        x0, y0, x1, y1 = box
        crop = img.crop((x0, y0, x1, y1))
        # distort→degrade→ocropy chain. degrade_line wants binary-ish input;
        # the chain handles the thresholding internally.
        aug = distort_line(crop, **DISTORT_KW)
        aug = degrade_line(aug, **DEGRADE_KW)
        aug = ocropy_degrade(aug, **OCROPY_KW)
        # Paste at original bbox top-left. Augmented size differs from bbox;
        # crop to bbox size so we don't overwrite neighboring lines.
        aw, ah = aug.size
        paste_w = min(aw, x1 - x0)
        paste_h = min(ah, y1 - y0)
        canvas.paste(aug.crop((0, 0, paste_w, paste_h)), (x0, y0))
        n += 1
    canvas.save(str(out_img_path))
    # Copy XML with new imageFilename
    shutil.copyfile(str(xml_path), str(out_xml_path))
    tree = ET.parse(str(out_xml_path))
    page = tree.getroot().find(f"{{{NS}}}Page")
    if page is not None:
        page.set("imageFilename", out_image_basename)
    tree.write(str(out_xml_path), xml_declaration=True, encoding="utf-8")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment 148-line Sogdian train set (§3).")
    ap.add_argument("--manifest", default="data/manifests/c2av-finetune.json")
    ap.add_argument("--out-dir", default="dataset/christian_sogdian_c2av/augmented")
    ap.add_argument("--out-manifest", default="data/manifests/c2av-finetune-aug.json")
    ap.add_argument("--variants", type=int, default=VARIANTS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
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
    for entry in train_entries:
        # Keep original entry unchanged
        new_train.append(entry)
        xml_rel = entry["xml_path"]
        xml_path = base_dir / xml_rel
        # Resolve image via XML imageFilename
        tree = ET.parse(str(xml_path))
        page = tree.getroot().find(f"{{{NS}}}Page")
        img_name = page.get("imageFilename")
        img_path = xml_path.parent / img_name
        ms_id = entry["manuscript_id"]
        for v in range(args.variants):
            seed = args.seed + hash((ms_id, v)) % (2**31)
            out_img = aug_dir / f"{ms_id}_v{v}.png"
            out_xml = aug_xml_dir / f"{ms_id}_v{v}.xml"
            n = augment_plate(
                img_path, xml_path, out_img, out_xml, out_img.name, seed
            )
            total_lines += n
            new_train.append(
                {
                    "id": f"{ms_id}-aug-v{v}",
                    "manuscript_id": ms_id,
                    "xml_path": str(out_xml.relative_to(Path(args.out_dir).parent)),
                    "augmented": True,
                }
            )

    aug_manifest = dict(manifest)
    aug_manifest["manifest_id"] = manifest["manifest_id"] + "-aug"
    aug_manifest["base_dir"] = str(Path(args.out_dir).parent / out_dir.parent)
    # ponytail: base_dir must resolve xml_path relative to itself. The augmented
    # XMLs live under {out_dir}/gt/ and reference images under {out_dir}/plates/.
    aug_manifest["base_dir"] = str(out_dir.parent)  # = dataset/christian_sogdian_c2av
    # Rewrite xml_path to be relative to base_dir (out_dir.parent)
    for e in new_train:
        if e.get("augmented"):
            e["xml_path"] = str((out_dir / "gt" / Path(e["xml_path"]).name).relative_to(out_dir.parent))
    aug_manifest["partitions"] = {
        "train": new_train,
        "validation": val_entries,
        "holdout": holdout_entries,
    }
    aug_manifest["metadata"] = dict(manifest.get("metadata", {}))
    aug_manifest["metadata"]["augmented_variants"] = args.variants
    aug_manifest["metadata"]["augmented_train_plates"] = len(new_train)
    aug_manifest["metadata"]["augmented_train_lines"] = total_lines + manifest["metadata"]["train_lines"]

    Path(args.out_manifest).write_text(json.dumps(aug_manifest, indent=2, ensure_ascii=False))
    print(
        f"§3 done: {len(new_train)} train plates ({len(train_entries)} orig + "
        f"{len(new_train) - len(train_entries)} aug), ~{total_lines} augmented lines. "
        f"Manifest → {args.out_manifest}"
    )


if __name__ == "__main__":
    main()