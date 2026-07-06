#!/usr/bin/env python3
"""Triangulate CER on c2av01(train)/c2av11(val)/c2av12(holdout) for a model.
Usage: triangulate_eval.py [MODEL]
Reuses harness polygon enrichment + KetosTrainer.test_model (kraken 7.0 API).
"""
from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from msocr.evaluation.harness import _parse_ketos_stdout
from msocr.training.ketos_trainer import KetosTrainer
from msocr.training.orchestrator import _enrich_xml_with_polygons, _resolve_image_for_xml

GT_DIR = ROOT / "dataset/christian_sogdian_c2av"
PLATES = [
    ("train", "c2av01", "gt/c2av01_page_01.xml"),
    ("val",   "c2av11", "gt/c2av11_page_11.xml"),
    ("holdout", "c2av12", "gt/c2av12_page_12.xml"),
]
TRAINER = KetosTrainer({
    "dataset": {"format_type": "page"},
    "model": {"spec": "placeholder"},
    "training": {"epochs": 0, "device": "cpu", "workers": 1},
    "output": {"model_prefix": "placeholder"},
})


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "models/kraken/c2av_finetune_200ep"
    model_path = str(Path(model).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"=== triangulation: {Path(model).name} ===")
        for partition, mid, xml_rel in PLATES:
            xml = GT_DIR / xml_rel
            image = Path(_resolve_image_for_xml(xml, None))
            enriched = tmp_path / f"{mid}_poly.xml"
            _enrich_xml_with_polygons(xml, image, enriched)
            shutil.copy2(image, tmp_path / image.name)
            stdout = TRAINER.test_model(model_path, str(enriched))
            m = _parse_ketos_stdout(stdout)
            print(f"  {partition:8s} {mid}  CER={m.get('cer')}  WER={m.get('wer')}  acc={m.get('accuracy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())