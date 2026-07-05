#!/usr/bin/env python3
"""0.1 Confusion-matrix audit — pull per-char confusion from ketos test on the
shipped 3.2 model against c2av12 holdout.

Ponytail: reuses the harness's _enrich_xml_with_polygons + _resolve_image_for_xml
helpers, then runs ketos test directly and captures the full report (the harness
discards everything except CER/WER/Accuracy).

Outputs:
  reports/confusion_c2av12_holdout.txt   — full ketos test rendered report
  reports/confusion_c2av12_holdout.json  — parsed counts table (top substitutions)
"""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from msocr.training.orchestrator import _enrich_xml_with_polygons, _resolve_image_for_xml

MODEL = ROOT / "models/kraken/c2av_finetune_union_frozen.safetensors"
XML = ROOT / "dataset/christian_sogdian_c2av/gt/c2av12_page_12.xml"
REPORTS = ROOT / "reports"


def main() -> int:
    if not MODEL.exists():
        print(f"missing model: {MODEL}", file=sys.stderr)
        return 1
    if not XML.exists():
        print(f"missing xml: {XML}", file=sys.stderr)
        return 1
    REPORTS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image = _resolve_image_for_xml(XML, None)
        enriched_xml = tmp_path / "c2av12_poly.xml"
        _enrich_xml_with_polygons(XML, image, enriched_xml)
        shutil.copy2(image, tmp_path / Path(image).name)

        cmd = [str(ROOT / ".venv/bin/ketos"), "test",
               "-m", str(MODEL),
               "-f", "page",
               str(enriched_xml)]
        print(f"running: {' '.join(cmd)}", file=sys.stderr)
        # ponytail: capture both stdout+stderr; the rendered report goes to stderr
        # (kraken uses `message()` which writes to rich console on stderr).
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = result.stdout + "\n--- STDERR ---\n" + result.stderr

    out_txt = REPORTS / "confusion_c2av12_holdout.txt"
    out_txt.write_text(combined)
    print(f"wrote {out_txt} ({len(combined)} bytes)", file=sys.stderr)

    # ponytail: parse the counts table from the rendered report.
    # Format (from kraken/templates/report):
    #   Correct | Generated | Errors
    #   <char>  | <char>    | <n>
    # plus per-script table.
    counts = _parse_counts(combined)
    out_json = REPORTS / "confusion_c2av12_holdout.json"
    out_json.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"wrote {out_json} ({len(counts)} substitution rows)", file=sys.stderr)

    # Summary to stdout
    print(f"\n=== c2av12 holdout confusion audit ===")
    print(f"total substitution types: {len(counts)}")
    if counts:
        print(f"top 15 substitutions (correct -> generated):")
        for row in counts[:15]:
            print(f"  {row['correct']!r:8} -> {row['generated']!r:8}  {row['errors']:4}")
    return 0


def _parse_counts(report: str) -> list[dict]:
    """Extract the confusion table from the rendered ketos report.

    kraken template format (report.txt):
        Errors<TAB>Correct-Generated
        39<TAB>{ ܝ } - {  }
    where { X } is the gt char and { Y } is the pred char. Empty = dropped/inserted.
    """
    counts: list[dict] = []
    in_table = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("Errors") and "Correct-Generated" in s:
            in_table = True
            continue
        if in_table:
            if not s:
                if counts:
                    break
                continue
            # ponytail: row format is "<int>\t{ a } - { b }"
            m = re.match(r"^(\d+)\s+\{\s*(.*?)\s*\}\s*-\s*\{\s*(.*?)\s*\}", s)
            if not m:
                continue
            errors = int(m.group(1))
            correct = m.group(2)
            generated = m.group(3)
            counts.append({"correct": correct, "generated": generated, "errors": errors})
    counts.sort(key=lambda x: x["errors"], reverse=True)
    return counts


if __name__ == "__main__":
    raise SystemExit(main())