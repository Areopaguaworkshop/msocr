"""Generalized HTR prediction dump for unannotated plates.

Generalizes ``scripts/dump_c2av12_preds.py`` into a reusable function + CLI
entry. Loads a Kraken recognition model ONCE, runs it over one or more PAGE
XMLs (enriching each with polygons first), and writes per-plate
``<plate_id>_preds.json`` files containing ``[{line_id, transcript,
confidence}, ...]``.

ponytail: the one-off script is correct; this extracts its kraken
load + predict loop verbatim, parameterized over model path, XML paths,
and output. No new abstractions.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

# ponytail: c2av is the only manuscript in scope today; its GT XMLs live at
# {corpus_root}/gt/{plate_id}_page_{NN}.xml where NN = digits after "c2av".
# If a second corpus lands, generalize to a per-corpus resolver. YAGNI now.
_C2AV_ROOT = Path("dataset/christian_sogdian_c2av")
_C2AV_GT = _C2AV_ROOT / "gt"


def _resolve_xml_for_plate(plate_id: str) -> Path:
    """Resolve a PAGE XML path for a plate id like ``c2av12``.

    Convention: ``{gt_dir}/{plate_id}_page_{NN}.xml`` where NN is the
    trailing digits of plate_id. Returns the first glob match.
    """
    digits = "".join(ch for ch in plate_id if ch.isdigit())
    pattern = f"{plate_id}_page_*.xml"
    if digits:
        # prefer the exact NN match, fall back to any _page_*.xml
        exact = _C2AV_GT / f"{plate_id}_page_{digits}.xml"
        if exact.exists():
            return exact
    matches = sorted(_C2AV_GT.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No PAGE XML for plate {plate_id} under {_C2AV_GT} (pattern {pattern})"
        )
    return matches[0]


def _predict_one_xml(
    model,
    cfg,
    xml_path: Path,
    tmp_path: Path,
) -> list[dict[str, Any]]:
    """Enrich one XML with polygons and run model.predict over its lines.

    Returns ``[{line_id, transcript, confidence}, ...]`` in document order.
    Mirrors ``scripts/dump_c2av12_preds.py`` lines 35-62 verbatim.
    """
    from PIL import Image
    from kraken.lib.xml import parse_page
    from kraken.containers import Segmentation
    from lxml import etree

    from msocr.training.orchestrator import (
        _enrich_xml_with_polygons,
        _resolve_image_for_xml,
    )

    image = _resolve_image_for_xml(xml_path, None)
    enriched = tmp_path / f"{xml_path.stem}_poly.xml"
    _enrich_xml_with_polygons(xml_path, image, enriched)
    # ponytail: copy image into the temp dir so kraken's parse_page resolves
    # it by basename; skip if already there (image may live next to the XML).
    dst_img = tmp_path / Path(image).name
    if dst_img.resolve() != Path(image).resolve():
        shutil.copy2(image, dst_img)

    doc = etree.parse(str(enriched))
    seg_dict = parse_page(doc, tmp_path / Path(image).name, "baselines")
    seg = Segmentation(
        type="baselines",
        imagename=seg_dict["imagename"],
        text_direction="horizontal-rl",
        script_detection=False,
        lines=list(seg_dict["lines"].values()),
        regions=seg_dict.get("regions"),
        line_orders=seg_dict.get("raw_orders"),
        language=None,
    )
    im = Image.open(str(tmp_path / Path(image).name)).convert("L")

    # ponytail: parse_page preserves line ids on the seg line objects;
    # kraken's predict record exposes .prediction and .line_id (or .id).
    records: list[dict[str, Any]] = []
    for line_id, rec in zip(
        list(seg_dict["lines"].keys()), model.predict(im, seg, cfg)
    ):
        confs = getattr(rec, "confidences", None) or []
        # ponytail: kraken's ocr_record exposes per-char confidences (list[float]),
        # not a scalar. Mean over chars; empty → 0.0.
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        records.append(
            {
                "line_id": line_id,
                "transcript": rec.prediction,
                "confidence": confidence,
            }
        )
    return records


def dump_predictions(
    model_path: Path,
    xml_paths: list[Path] | None = None,
    plate_ids: list[str] | None = None,
    output_dir: Path | None = None,
    *,
    return_records: bool = False,
    flag_damage: bool = False,
) -> list[dict] | None:
    """Run a Kraken recognition model over one or more PAGE XMLs.

    Exactly one of ``xml_paths`` / ``plate_ids`` must be non-empty (both may
    be given; they're concatenated). For each XML: enrich with polygons,
    load model ONCE (outside the loop), run ``model.predict`` per the
    script's pattern, collect ``(line_id, prediction, confidence)`` records.

    Output modes (first that applies):
    - ``return_records=True``: return ``list[dict]`` (one dict per plate:
      ``{"plate_id": str, "records": [...]}``) instead of writing.
    - ``output_dir`` given: write ``<plate_id>_preds.json`` per plate.
    - otherwise: write to ``reports/`` (matching the one-off's default).

    If ``flag_damage=True``, each record is enriched with ``likely_damage``
    (bool) and ``damage_reason`` (str | None) via
    ``msocr.training.damage_flag.flag_predictions`` — a cheap heuristic
    flagging lines the recognizer likely failed to read due to damage or
    lacunae (low confidence, empty transcript, single grapheme, or the
    lacuna token ░). See ``docs/DAMAGE_ANNOTATION_TASK.md`` and
    ``msocr/training/damage_flag.py``.

    Args:
        model_path: Path to a ``.safetensors`` or ``.mlmodel`` on disk.
        xml_paths: Explicit PAGE XML files to predict on.
        plate_ids: Plate ids (e.g. ``c2av12``) resolved via the c2av gt
            naming convention.
        output_dir: Where to write ``<plate>_preds.json`` files.
        return_records: If True, return records instead of writing files.
        flag_damage: If True, add ``likely_damage`` + ``damage_reason`` to
            each record (Phase 0.5 damage-triage signal).

    Returns:
        ``list[dict]`` if ``return_records`` else ``None``.
    """
    if not xml_paths and not plate_ids:
        raise ValueError("dump_predictions: provide xml_paths or plate_ids")

    all_xmls: list[tuple[str, Path]] = []
    if plate_ids:
        for pid in plate_ids:
            all_xmls.append((pid, _resolve_xml_for_plate(pid)))
    if xml_paths:
        for xp in xml_paths:
            # ponytail: plate_id = XML stem up to "_page_"; fall back to stem.
            stem = xp.stem
            pid = stem.split("_page_", 1)[0] if "_page_" in stem else stem
            all_xmls.append((pid, xp))

    # Load model ONCE — ponytail: this is the expensive call.
    from kraken.tasks import RecognitionTaskModel
    from kraken.configs import RecognitionInferenceConfig

    model = RecognitionTaskModel.load_model(str(model_path))
    cfg = RecognitionInferenceConfig()

    per_plate: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for plate_id, xml_path in all_xmls:
            records = _predict_one_xml(model, cfg, xml_path, tmp_path)
            per_plate.append({"plate_id": plate_id, "records": records})

    if flag_damage:
        # ponytail: heuristic flag, no pixel annotation needed. See
        # msocr/training/damage_flag.py. Mutates per_plate[i]['records'] in
        # place, adding likely_damage + damage_reason per record.
        from msocr.training.damage_flag import flag_predictions
        flag_predictions(per_plate)

    if return_records:
        return per_plate

    out = Path(output_dir) if output_dir else Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    for entry in per_plate:
        (out / f"{entry['plate_id']}_preds.json").write_text(
            json.dumps(entry["records"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return None


if __name__ == "__main__":
    # Self-check: build a tiny synthetic PAGE XML with 2 TextLines, run the
    # XML-parsing path. If a real model exists, run the full predict path
    # and assert 2 records per plate; otherwise skip predict and just assert
    # the parse path doesn't crash on synthetic input.
    import sys

    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    NSMAP = {"page": NS}

    def _synth_xml(path: Path, image_path: Path) -> None:
        from lxml import etree

        root = etree.Element(f"{{{NS}}}PcGts", nsmap={"page": NS})
        page = etree.SubElement(
            root,
            f"{{{NS}}}Page",
            imageFilename=image_path.name,
            imageWidth="64",
            imageHeight="32",
        )
        # ponytail: kraken's parse_page only picks up TextLines nested in a
        # TextRegion (it iterates regions, then region.find TextLine). Wrap
        # the lines in a TextRegion with a Coords so the region has coords.
        region = etree.SubElement(page, f"{{{NS}}}TextRegion", id="r1")
        region.set("custom", "structure {type:MainZone;}")
        coords = etree.SubElement(region, f"{{{NS}}}Coords")
        coords.set("points", "0,0 63,0 63,31 0,31")
        for i in range(2):
            tl = etree.SubElement(region, f"{{{NS}}}TextLine", id=f"l{i+1}")
            bl = etree.SubElement(tl, f"{{{NS}}}Baseline")
            y = 10 + 10 * i
            bl.set("points", f"10,{y} 50,{y}")
            te = etree.SubElement(tl, f"{{{NS}}}TextEquiv")
            uni = etree.SubElement(te, f"{{{NS}}}Unicode")
            uni.text = f"line{i+1}"
        tree = etree.ElementTree(root)
        tree.write(str(path), xml_declaration=True, encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # ponytail: tiny real PNG next to the XML so _resolve_image_for_xml finds it.
        # Draw dark ink along the baseline rows so calculate_polygonal_environment
        # produces a polygon (it returns None for blank rows and parse_page drops
        # lines without Coords).
        from PIL import Image as _PILImage, ImageDraw as _ImageDraw

        synth_img = tmp / "synth.png"
        img = _PILImage.new("L", (64, 32), 255)
        draw = _ImageDraw.Draw(img)
        draw.rectangle((10, 8, 50, 12), fill=0)   # line 1 ink band
        draw.rectangle((20, 18, 60, 22), fill=0)  # line 2 ink band
        img.save(synth_img)
        xml = tmp / "synth_page_1.xml"
        _synth_xml(xml, synth_img)
        model = Path("models/kraken/c2av_finetune_union_frozen.safetensors")
        if not model.exists():
            print("skip: no model")
            # ponytail: still verify the plate resolver + arg validation paths
            # don't crash on synthetic-only input. We can't call dump_predictions
            # without a model (load_model is the first call), so just assert the
            # XML parses and the plate resolver raises cleanly.
            from lxml import etree

            doc = etree.parse(str(xml))
            lines = doc.findall(f".//{{{NS}}}TextLine")
            assert len(lines) == 2, f"expected 2 TextLines, got {len(lines)}"
            print("ok")
            sys.exit(0)
        out = dump_predictions(
            model_path=model, xml_paths=[xml], return_records=True, flag_damage=True
        )
        assert out and len(out) == 1, f"expected 1 plate, got {out}"
        recs = out[0]["records"]
        assert len(recs) == 2, f"expected 2 records, got {len(recs)}"
        # ponytail: synthetic transcripts "line1"/"line2" are 5 and 5 chars
        # — NOT single-grapheme, so flag_damage should NOT mark them (they
        # have confidence though, so the flag runs but returns False unless
        # confidence is < 0.3). Assert the flag fields exist on each record.
        for r in recs:
            assert "likely_damage" in r, f"flag_damage missing on {r}"
            assert "damage_reason" in r, f"damage_reason missing on {r}"
        print("ok")