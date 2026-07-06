#!/usr/bin/env python3
"""Dump HTR predictions on c2av12 holdout as Syriac + Latin (for 3.1 lexicon work).

Reuses harness polygon enrichment, runs model.predict, writes:
  reports/c2av12_pred_syriac.txt  — raw model output (Syriac)
  reports/c2av12_pred_latin.txt   — via syriac_to_latin.py
  reports/c2av12_gt_latin.txt     — ground truth Latin (for diff)
"""
from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from kraken.tasks import RecognitionTaskModel
from kraken.lib.xml import parse_page
from kraken.containers import Segmentation

from msocr.training.orchestrator import _enrich_xml_with_polygons, _resolve_image_for_xml
import scripts.syriac_to_latin as s2l

MODEL = ROOT / "models/kraken/c2av_finetune_union_frozen.safetensors"
XML = ROOT / "dataset/christian_sogdian_c2av/gt/c2av12_page_12.xml"
REPORTS = ROOT / "reports"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image = _resolve_image_for_xml(XML, None)
        enriched = tmp_path / "c2av12_poly.xml"
        _enrich_xml_with_polygons(XML, image, enriched)
        shutil.copy2(image, tmp_path / Path(image).name)

        model = RecognitionTaskModel.load_model(str(MODEL))
        from kraken.configs import RecognitionInferenceConfig
        cfg = RecognitionInferenceConfig()
        # ponytail: parse_page returns an XMLPage; its .lines gives Segmentation-like.
        # Easier: use kraken's page parser to build a Segmentation directly.
        from kraken.containers import Segmentation
        from lxml import etree
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
        preds: list[str] = []
        for record in model.predict(im, seg, cfg):
            preds.append(record.prediction)

    (REPORTS / "c2av12_pred_syriac.txt").write_text("\n".join(preds) + "\n")
    latin = [s2l.convert(p) for p in preds]
    (REPORTS / "c2av12_pred_latin.txt").write_text("\n".join(latin) + "\n")

    # GT Latin: extract from the XML TextLine text and convert
    from lxml import etree
    doc = etree.parse(str(XML))
    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    gt_lines = []
    for tl in doc.iter(f"{{{NS}}}TextLine"):
        te = tl.find(f"{{{NS}}}TextEquiv")
        if te is not None:
            uni = te.find(f"{{{NS}}}Unicode")
            if uni is not None and uni.text:
                gt_lines.append(s2l.convert(uni.text))
    (REPORTS / "c2av12_gt_latin.txt").write_text("\n".join(gt_lines) + "\n")

    print(f"pred syriac: {len(preds)} lines → {REPORTS/'c2av12_pred_syriac.txt'}")
    print(f"pred latin:  {len(latin)} lines → {REPORTS/'c2av12_pred_latin.txt'}")
    print(f"gt latin:    {len(gt_lines)} lines → {REPORTS/'c2av12_gt_latin.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())