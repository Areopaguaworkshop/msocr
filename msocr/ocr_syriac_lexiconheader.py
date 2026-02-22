#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

SYR_START = 0x0700
SYR_END = 0x074F


def keep_syriac(text: str) -> str:
    kept = []
    prev_space = False
    for ch in text:
        cp = ord(ch)
        if SYR_START <= cp <= SYR_END:
            kept.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space:
                kept.append(" ")
                prev_space = True
    return "".join(kept).strip()


def detect_sides(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    y0 = max(0, int(height * 0.30))
    band = gray[y0:, :]
    bw = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = []
    band_h = height - y0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8 or w < 2 or h < 3:
            continue
        if w > 0.33 * width and h < 0.22 * height:
            continue
        if h > 0.88 * band_h and w < 0.04 * width:
            continue
        gx, gy = int(x), int(y + y0)
        cx = gx + w / 2
        comps.append((gx, gy, int(w), int(h), int(area), cx))

    out = {"left": None, "right": None}
    for side, lo, hi in (("left", 0, 0.47 * width), ("right", 0.53 * width, width)):
        cand = [c for c in comps if lo <= c[5] <= hi]
        if not cand:
            continue
        xs = [c[0] for c in cand]
        ys = [c[1] for c in cand]
        x2 = [c[0] + c[2] for c in cand]
        y2 = [c[1] + c[3] for c in cand]
        pad = max(3, int(0.01 * width))
        x = max(0, min(xs) - pad)
        y = max(0, min(ys) - pad)
        xx = min(width, max(x2) + pad)
        yy = min(height, max(y2) + pad)
        bwid = xx - x
        bhei = yy - y
        if bwid < 12 or bhei < 10:
            continue
        out[side] = [int(x), int(y), int(bwid), int(bhei)]
    return out


def ocr_crop(img, bbox):
    x, y, w, h = bbox
    crop = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    proc = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        cv2.imwrite(tmp_name, proc)
        txt = subprocess.check_output(
            ["tesseract", tmp_name, "stdout", "-l", "syr", "--psm", "7"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        tsv = subprocess.check_output(
            ["tesseract", tmp_name, "stdout", "-l", "syr", "--psm", "7", "tsv"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        txt = ""
        tsv = ""
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    confs = []
    for line in tsv.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 12:
            try:
                conf = float(parts[10])
            except ValueError:
                continue
            if conf >= 0 and parts[11].strip():
                confs.append(conf)
    mean_conf = round((sum(confs) / len(confs)) / 100.0, 4) if confs else 0.0

    raw = " ".join(txt.split())
    raw = unicodedata.normalize("NFC", raw)
    syr = keep_syriac(raw)
    return raw, syr, mean_conf


def process_file(path: Path, conf_threshold: float):
    img = cv2.imread(str(path))
    match = re.search(r"(\d+)", path.stem)
    page_num = int(match.group(1)) if match else None

    rec = {
        "page_number": page_num,
        "file_name": path.name,
        "left": {"bbox": None, "text_raw": "", "text_syriac": "", "confidence": 0.0},
        "right": {"bbox": None, "text_raw": "", "text_syriac": "", "confidence": 0.0},
        "flags": [],
    }
    if img is None:
        rec["flags"].append("image_read_error")
        return rec

    sides = detect_sides(img)
    for side in ("left", "right"):
        bbox = sides.get(side)
        if bbox is None:
            rec["flags"].append(f"missing_{side}_region")
            continue
        raw, syr, conf = ocr_crop(img, bbox)
        rec[side] = {
            "bbox": bbox,
            "text_raw": raw,
            "text_syriac": syr,
            "confidence": conf,
        }
        if not syr:
            rec["flags"].append(f"empty_{side}_text")
        if conf < conf_threshold:
            rec["flags"].append(f"low_{side}_confidence")
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="OCR Syriac text from left/right sides of lexicon header images."
    )
    parser.add_argument(
        "--input-dir",
        default="models/SyriacLexiconheader",
        help="Directory containing page_*.png files.",
    )
    parser.add_argument(
        "--output-json",
        default="output/syriac_lexiconheader_ocr.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write output as JSONL (one JSON object per line) instead of JSON array.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent OCR workers.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for low_* flags.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N pages (0 = all).",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        in_dir.glob("page_*.png"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if args.limit > 0:
        files = files[: args.limit]

    total = len(files)
    print(f"files={total} workers={args.workers}", flush=True)

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(process_file, p, args.conf_threshold) for p in files]
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 100 == 0 or i == total:
                print(f"processed {i}/{total}", flush=True)

    records.sort(
        key=lambda r: (
            r["page_number"] if isinstance(r.get("page_number"), int) else 10**9
        )
    )
    with out_path.open("w", encoding="utf-8") as f:
        if args.jsonl:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

    both_nonempty = sum(
        1 for r in records if r["left"]["text_syriac"] and r["right"]["text_syriac"]
    )
    low_conf_pages = sum(
        1 for r in records if any(flag.startswith("low_") for flag in r["flags"])
    )
    print(f"output={out_path}", flush=True)
    print(f"records={len(records)}", flush=True)
    print(f"both_nonempty={both_nonempty}", flush=True)
    print(f"low_conf_pages={low_conf_pages}", flush=True)


if __name__ == "__main__":
    main()
