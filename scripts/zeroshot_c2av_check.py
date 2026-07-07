#!/usr/bin/env python3
"""v6 §2.2 zero-shot backbone comparison.

Compares two candidate base models on plain-Syriac-only lines of the c2av
manuscript (12 plates: 10 train + 1 val + 1 holdout). Whichever has lower CER
becomes the Stage 2 fine-tune base. Pure inference + scoring, no retraining.

Approach: reuse the proven parse_page + model.predict(im, seg, cfg) path from
scripts/dump_c2av12_preds.py. Each plate's <Coords> are computed once via
kraken's calculate_polygonal_environment (orchestrator helper), then both
models run on the SAME Segmentation so any segmentation noise cancels out.

Line filtering: keep a line iff set(gt_text) <= set(base_model_codec.chars).
This drops lines containing any of the 5 Sogdian-specific codepoints (rows
37-41) and any char the base model never saw — a cleaner like-for-like
comparison. The 5 Sogdian chars: U+0741 qushshaya, U+0742 rukkakha,
U+074D sogdian zhain, U+074E sogdian khaph, U+074F sogdian fe.

Usage: uv run python3 scripts/zeroshot_c2av_check.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lxml import etree  # noqa: E402
from PIL import Image  # noqa: E402

from kraken.configs import RecognitionInferenceConfig  # noqa: E402
from kraken.containers import Segmentation  # noqa: E402
from kraken.lib.xml import parse_page  # noqa: E402
from kraken.tasks import RecognitionTaskModel  # noqa: E402

from msocr.training.orchestrator import _enrich_xml_with_polygons  # noqa: E402

# --- config ----------------------------------------------------------------
MANIFEST = ROOT / "data/manifests/c2av-finetune.json"
MODEL_A_PATH = ROOT / "models/kraken/sophro_mhiro_syriac.safetensors"
MODEL_B_PATH = ROOT / "models/kraken/sophro_jer36_adapted.safetensors"
REPORTS_DIR = ROOT / "reports"
RESULTS_JSON = REPORTS_DIR / "zeroshot_c2av_check.json"

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS = f"{{{PAGE_NS}}}"


# --- stdlib Levenshtein (no new dep; mirrors scripts/cer.py) ----------------
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


# --- XML / image plumbing ---------------------------------------------------
def _resolve_image(xml_path: Path) -> Path:
    """Image lives at the absolute path in <Page imageFilename>; fall back to
    plates/<basename>.png next to the dataset root."""
    doc = etree.parse(str(xml_path))
    root = doc.getroot()
    page = root.find(f"{NS}Page")
    if page is None:
        raise ValueError(f"No <Page> in {xml_path}")
    fname = page.get("imageFilename", "")
    if fname and Path(fname).exists():
        return Path(fname)
    # fall back: dataset/.../plates/p-NN.png derived from plate number in xml name
    stem_num = xml_path.stem.split("_page_")[1] if "_page_" in xml_path.stem else xml_path.stem
    cand = xml_path.parent.parent / "plates" / f"p-{stem_num}.png"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"image for {xml_path} not found (tried {fname}, {cand})")


def _extract_gt_lines(xml_path: Path) -> list[tuple[str, str]]:
    """Return [(line_id, gt_text)] in document order. line_id = TextLine id attr
    or 'L<index>'."""
    doc = etree.parse(str(xml_path))
    root = doc.getroot()
    page = root.find(f"{NS}Page")
    out: list[tuple[str, str]] = []
    for i, tl in enumerate(page.iter(f"{NS}TextLine")):
        te = tl.find(f"{NS}TextEquiv")
        text = ""
        if te is not None:
            uni = te.find(f"{NS}Unicode")
            if uni is not None and uni.text:
                text = uni.text
        lid = tl.get("id") or f"L{i}"
        out.append((lid, text))
    return out


# --- model helpers ----------------------------------------------------------
def load_codec_chars(model_path: Path) -> set[str]:
    """Return the set of output chars the model's codec can decode to.
    codec.c2l maps char -> label; keys are the chars (space included).
    row 0 is the CTC blank and is NOT in c2l keys."""
    m = RecognitionTaskModel.load_model(str(model_path))
    return set(m.net.codec.c2l.keys())


def predict_plate(model, image: Path, enriched_xml: Path, tmp_path: Path) -> list[str]:
    """Run model on one plate, return predictions in line order.
    parse_page(doc, filename, linetype): doc must be a parsed lxml tree,
    filename is a Path (it does filename.parent to resolve imageFilename).
    The XML's imageFilename is an absolute path; joinpath(absolute) returns
    the absolute path, so passing tmp_path as filename works because the
    image is already copied there AND the absolute path resolves directly."""
    doc = etree.parse(str(enriched_xml))
    filename = tmp_path / image.name
    seg_dict = parse_page(doc, filename, "baselines")
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
    # imagename is an absolute Path to the real plate image; open that.
    im = Image.open(str(seg.imagename)).convert("L")
    return [rec.prediction for rec in model.predict(im, seg, RecognitionInferenceConfig())]


# --- main -------------------------------------------------------------------
def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base_dir = ROOT / manifest["base_dir"]
    plates = []
    for part in ("train", "validation", "holdout"):
        for entry in manifest["partitions"].get(part, []):
            plates.append(
                {
                    "id": entry["id"],
                    "part": part,
                    "manuscript_id": entry["manuscript_id"],
                    "xml_path": base_dir / entry["xml_path"],
                }
            )
    plates.sort(key=lambda p: (p["part"], p["manuscript_id"]))

    print(f"Loading base codec from {MODEL_A_PATH.name} ...")
    base_chars = load_codec_chars(MODEL_A_PATH)
    # ponytail: include space + base consonants; the 5 Sogdian chars (U+0741,
    # U+0742, U+074D, U+074E, U+074F) are NOT in base_chars, so any line using
    # them is dropped. Also drops lines with chars the base model never saw.
    print(f"  base codec chars ({len(base_chars)}): {''.join(sorted(base_chars))!r}")

    print(f"Loading Model A: {MODEL_A_PATH.name} ...")
    model_a = RecognitionTaskModel.load_model(str(MODEL_A_PATH))
    print(f"Loading Model B: {MODEL_B_PATH.name} ...")
    model_b = RecognitionTaskModel.load_model(str(MODEL_B_PATH))
    print()

    # per-plate aggregation
    plate_results = []
    total_a_err = total_b_err = total_chars = 0
    n_lines_kept = 0
    n_lines_total = 0
    disagreements: list[dict] = []

    t0 = time.time()
    for plate in plates:
        xml_path = plate["xml_path"]
        if not xml_path.exists():
            print(f"[skip] {plate['id']}: XML missing {xml_path}")
            continue
        gt_lines = _extract_gt_lines(xml_path)
        n_lines_total += len(gt_lines)

        # build a single enriched XML once per plate (segmentation cancels out)
        try:
            image = _resolve_image(xml_path)
        except FileNotFoundError as e:
            print(f"[skip] {plate['id']}: {e}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            enriched = tmp_path / f"{xml_path.stem}_poly.xml"
            try:
                _enrich_xml_with_polygons(xml_path, image, enriched)
            except Exception as e:
                print(f"[skip] {plate['id']}: polygon enrichment failed: {e}")
                continue
            shutil.copy2(image, tmp_path / image.name)

            # filter GT lines to plain-Syriac-only BEFORE predicting
            # (predict returns ALL lines in seg order; we zip with gt and mask)
            try:
                preds_a = predict_plate(model_a, image, enriched, tmp_path)
                preds_b = predict_plate(model_b, image, enriched, tmp_path)
            except Exception as e:
                print(f"[skip] {plate['id']}: prediction failed: {e}")
                continue

        # align by index (parse_page preserves document order == gt order)
        kept_a_err = kept_b_err = kept_chars = 0
        kept_n = 0
        for i, (lid, gt) in enumerate(gt_lines):
            pred_a = preds_a[i] if i < len(preds_a) else ""
            pred_b = preds_b[i] if i < len(preds_b) else ""
            if not set(gt) <= base_chars:
                continue
            if pred_a == "" and pred_b == "":
                # both models blanked the line — still count it (it's a real
                # error signal), but guard against total silence skewing CER
                pass
            kept_n += 1
            ea = levenshtein(pred_a, gt)
            eb = levenshtein(pred_b, gt)
            kept_a_err += ea
            kept_b_err += eb
            kept_chars += len(gt)
            if len(disagreements) < 10 and ea != eb:
                disagreements.append(
                    {
                        "plate": plate["id"],
                        "line_id": lid,
                        "gt": gt,
                        "pred_a": pred_a,
                        "pred_b": pred_b,
                        "cer_a": round(100 * ea / max(len(gt), 1), 1),
                        "cer_b": round(100 * eb / max(len(gt), 1), 1),
                    }
                )

        total_a_err += kept_a_err
        total_b_err += kept_b_err
        total_chars += kept_chars
        n_lines_kept += kept_n
        cer_a = 100 * kept_a_err / max(kept_chars, 1)
        cer_b = 100 * kept_b_err / max(kept_chars, 1)
        winner = "A" if cer_a < cer_b else ("B" if cer_b < cer_a else "tie")
        plate_results.append(
            {
                "plate": plate["id"],
                "part": plate["part"],
                "manuscript_id": plate["manuscript_id"],
                "n_lines_total": len(gt_lines),
                "n_lines_kept": kept_n,
                "chars": kept_chars,
                "a_err": kept_a_err,
                "b_err": kept_b_err,
                "cer_a_pct": round(cer_a, 2),
                "cer_b_pct": round(cer_b, 2),
                "winner": winner,
            }
        )
        print(
            f"{plate['id']:<16} kept {kept_n:3d}/{len(gt_lines):3d} "
            f"chars {kept_chars:5d}  A={cer_a:5.2f}%  B={cer_b:5.2f}%  -> {winner}"
        )

    elapsed = time.time() - t0
    cer_a_total = 100 * total_a_err / max(total_chars, 1)
    cer_b_total = 100 * total_b_err / max(total_chars, 1)
    diff = cer_a_total - cer_b_total
    if abs(diff) < 1.0:
        verdict = f"Tie (within {abs(diff):.2f} pts)"
    elif diff < 0:
        verdict = "Base wins"
    else:
        verdict = "Jer36-adapted wins"

    print("\n" + "=" * 70)
    print(f"Lines total: {n_lines_total}   kept: {n_lines_kept}   chars: {total_chars}")
    print(f"Model A (base)            CER: {cer_a_total:.2f}%  ({total_a_err}/{total_chars})")
    print(f"Model B (jer36-adapted)    CER: {cer_b_total:.2f}%  ({total_b_err}/{total_chars})")
    print(f"Delta (A - B):            {diff:+.2f} pts")
    print(f"Verdict: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    print("\nPer-plate breakdown:")
    print(f"{'plate':<16} {'part':<9} {'kept':<5} {'chars':<6} {'A CER':>8} {'B CER':>8} {'winner':>6}")
    for pr in plate_results:
        print(
            f"{pr['plate']:<16} {pr['part']:<9} {pr['n_lines_kept']:<5} "
            f"{pr['chars']:<6} {pr['cer_a_pct']:>7.2f}% {pr['cer_b_pct']:>7.2f}% {pr['winner']:>6}"
        )

    print("\nDisagreement examples (A err != B err):")
    for d in disagreements[:10]:
        print(f"  [{d['plate']} {d['line_id']}] A={d['cer_a']}% B={d['cer_b']}%")
        print(f"    GT : {d['gt']!r}")
        print(f"    A  : {d['pred_a']!r}")
        print(f"    B  : {d['pred_b']!r}")

    results = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "model_a": {"name": MODEL_A_PATH.name, "path": str(MODEL_A_PATH.relative_to(ROOT))},
        "model_b": {"name": MODEL_B_PATH.name, "path": str(MODEL_B_PATH.relative_to(ROOT))},
        "n_lines_total": n_lines_total,
        "n_lines_kept": n_lines_kept,
        "total_chars": total_chars,
        "cer_a_pct": round(cer_a_total, 3),
        "cer_b_pct": round(cer_b_total, 3),
        "delta_a_minus_b_pts": round(diff, 3),
        "verdict": verdict,
        "base_codec_chars": sorted(base_chars),
        "elapsed_sec": round(elapsed, 1),
        "plates": plate_results,
        "disagreement_examples": disagreements[:10],
    }
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())