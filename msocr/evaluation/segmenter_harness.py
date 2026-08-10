"""Model-neutral segmenter evaluation harness (Phase 2).

Evaluates a line-level segmentation model against reviewed PAGE XML ground
truth on a frozen held-out fragment set. Produces a single JSON report per
(model, held-out-set) pair with six metrics:

  1. baseline precision / recall / F1          (layout_metrics.evaluate_layout)
  2. split_count / merge_count                 (layout_metrics.evaluate_layout)
  3. row-grouping accuracy                     (layout_metrics.evaluate_layout)
  4. gap-localization IoU                      (layout_metrics.evaluate_layout)
  5. reading-order pair accuracy               (layout_metrics.evaluate_layout)
  6. end-to-end CER                            (metrics.cer via recognizer)

The harness is architecture-independent: it accepts predictions in a unified
``layout.json`` dict format and converts to the ``evaluate_layout`` input
shape. Three adapter functions produce that format from the three candidate
backends (Kraken Segmentation, orli, YOLO-OBB). The adapters live here, not
in the segmenter modules, so the harness stays the single evaluation surface.

Ponytail: this is rung 4 — reuse ``evaluate_layout`` and ``cer`` that already
exist. No new metrics invented. The harness is glue + adapters + a runner.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from msocr.evaluation.layout_metrics import evaluate_layout
from msocr.evaluation.metrics import cer

logger = logging.getLogger(__name__)

# ── Unified prediction format ───────────────────────────────────────────────
# The harness accepts/produces this shape from any backend:
#   {"image": str, "lines": [{"id": str, "baseline": [[x,y],...],
#                             "polygon": [[x,y],...], "bbox": [l,t,r,b],
#                             "confidence": float}]}
# ─────────────────────────────────────────────────────────────────────────────


# ── Backend adapters: native output → unified layout.json ───────────────────


def kraken_segmentation_to_layout(seg: Any, image_path: str) -> dict[str, Any]:
    """Convert a ``kraken.containers.Segmentation`` to the unified layout dict.

    Works for BLLA, D-FINE, and orli (orli returns a ``Segmentation`` natively
    via its Kraken plugin integration). Reads ``baseline``, ``boundary``, and
    ``id`` from each line record; derives ``bbox`` and ``polygon`` from the
    boundary polygon when present.
    """
    lines = []
    for line in getattr(seg, "lines", []):
        baseline = getattr(line, "baseline", None)
        boundary = getattr(line, "boundary", None)
        entry: dict[str, Any] = {
            "id": getattr(line, "id", None),
            "baseline": [list(p) for p in baseline] if baseline else None,
            "polygon": [list(p) for p in boundary] if boundary else None,
        }
        if boundary:
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            entry["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        else:
            entry["bbox"] = None
        lines.append(entry)
    return {"image": image_path, "lines": lines}


def yolo_obb_to_layout(results: Any, image_path: str) -> dict[str, Any]:
    """Convert Ultralytics YOLO-OBB results to the unified layout dict.

    ``results`` is the list returned by ``model(image)``. Each result's
    ``.obb`` carries ``xyxyxyxy`` (N,4,2) corners, ``conf``, and ``cls``.
    The bottom edge of the rotated rect serves as the baseline (RTL: ordered
    bottom-right → bottom-left). The four corners form the polygon.
    """
    import numpy as np  # local: only YOLO callers pay the import cost

    def _to_np(t):
        # ponytail: Ultralytics returns torch tensors, but duck-type so the
        # adapter also works with plain numpy arrays (e.g. in unit tests).
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    lines = []
    for result in results:
        obb = getattr(result, "obb", None)
        if obb is None:
            continue
        corners_all = _to_np(obb.xyxyxyxy)  # (N, 4, 2)
        confs = _to_np(obb.conf)
        names = result.names
        classes = _to_np(obb.cls).astype(int)
        for i in range(len(corners_all)):
            c = corners_all[i]  # (4, 2) clockwise from top-left
            # ponytail: bottom edge as baseline. Corners are clockwise from
            # top-left → corners[1]=top-right, [2]=bottom-right, [3]=bottom-left.
            # RTL reading order: right→left, so baseline = [c[2], c[3]].
            baseline = [c[2].tolist(), c[3].tolist()]
            polygon = [c[j].tolist() for j in range(4)] + [c[0].tolist()]
            xs, ys = c[:, 0], c[:, 1]
            lines.append({
                "id": f"line_{i}",
                "baseline": baseline,
                "polygon": polygon,
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max())],
                "confidence": float(confs[i]),
                "class": names[int(classes[i])] if names else None,
            })
    return {"image": image_path, "lines": lines}


# ── GT adapter: PAGE XML v2 state → evaluate_layout ground-truth shape ──────


def gt_state_to_layout_dict(gt_state: dict[str, Any]) -> dict[str, Any]:
    """Convert ``parse_page_xml_to_v2`` output to the ``evaluate_layout`` shape.

    ``parse_page_xml_to_v2`` returns ``{"regions": [...], "lines": [...]}``
    with each line carrying ``baseline`` (list of (x,y) tuples), ``boundary``
    (polygon), ``id``, ``rowId`` (optional), and ``gapAfter`` (parsed into
    ``gaps`` at the top level). ``evaluate_layout`` expects the same
    ``{"lines": [...], "gaps": [...]}`` shape with ``baseline`` as a list of
    [x,y] lists. This function normalizes tuple→list and renames
    ``boundary``→``polygon`` for clarity (the metric code reads ``baseline``
    directly).
    """
    lines = []
    for line in gt_state.get("lines", []):
        baseline = line.get("baseline") or []
        polygon = line.get("boundary") or []
        entry: dict[str, Any] = {
            "id": line.get("id"),
            "baseline": [list(p) for p in baseline] if baseline else [],
            "polygon": [list(p) for p in polygon] if polygon else [],
        }
        if line.get("rowId"):
            entry["rowId"] = line["rowId"]
        lines.append(entry)
    gaps = []
    for gap in gt_state.get("gaps", []):
        poly = gap.get("polygon") or []
        gaps.append({"polygon": [list(p) for p in poly] if poly else [],
                     "afterLineId": gap.get("afterLineId"),
                     "rowId": gap.get("rowId")})
    return {"lines": lines, "gaps": gaps}


# ── End-to-end CER: segmenter + recognizer on one held-out image ────────────


def _end_to_end_cer(
    image_path: Path,
    layout: dict[str, Any],
    gt_state: dict[str, Any],
    recognizer_model_path: str,
    *,
    device: str = "cpu",
) -> float | None:
    """Run the recognizer on segmenter-predicted lines and compute CER vs GT.

    Matches predicted lines to GT lines by baseline proximity (one-to-one,
    symmetric mean distance ≤ 15px, same matching as ``evaluate_layout``),
    then runs Kraken recognition on each predicted line crop and compares
    the predicted text to the matched GT transcript using ``metrics.cer``.

    Returns ``None`` if no lines match or the recognizer is unavailable.
    """
    try:
        from kraken.tasks import RecognitionTaskModel
        from kraken.configs import RecognitionInferenceConfig
        from kraken.containers import Segmentation, BaselineLine
        from PIL import Image
    except ImportError:
        logger.warning("Kraken not available; skipping end-to-end CER")
        return None

    pred_lines = layout.get("lines", [])
    gt_lines = gt_state.get("lines", [])
    if not pred_lines or not gt_lines:
        return None

    # Build a Kraken Segmentation from the predicted baselines so the
    # recognizer can crop+recognize each line in one predict() call.
    kraken_lines = []
    for pl in pred_lines:
        baseline = pl.get("baseline") or []
        if not baseline:
            continue
        kraken_lines.append(BaselineLine(
            id=pl.get("id") or f"line_{len(kraken_lines)}",
            baseline=[(int(p[0]), int(p[1])) for p in baseline],
            boundary=[(int(p[0]), int(p[1])) for p in pl["polygon"]]
            if pl.get("polygon") else None,
        ))
    if not kraken_lines:
        return None

    im = Image.open(image_path).convert("RGB")
    seg = Segmentation(
        type="baselines",
        imagename=str(image_path),
        text_direction="horizontal-rl",
        script_detection=False,
        lines=kraken_lines,
        regions=None,
        line_orders=None,
        language=None,
    )
    rec_model = RecognitionTaskModel.load_model(recognizer_model_path)
    records = rec_model.predict(
        im, segmentation=seg,
        config=RecognitionInferenceConfig(device=device),
    )
    pred_text_by_id = {r.id: r.prediction for r in records}

    # Match predicted to GT by baseline proximity (reuse evaluate_layout's
    # matching by running it once and reading the matches list).
    gt_dict = gt_state_to_layout_dict(gt_state)
    layout_eval = evaluate_layout(gt_dict, layout)
    matches = layout_eval.get("matches", [])

    gt_by_id = {ln.get("id"): ln for ln in gt_lines}
    cer_values: list[float] = []
    for m in matches:
        gt_line = gt_by_id.get(m.get("ground_truth_id"))
        pred_text = pred_text_by_id.get(m.get("prediction_id"))
        if gt_line and pred_text is not None:
            ref = gt_line.get("transcript") or ""
            if ref:
                cer_values.append(cer(ref, pred_text))
    if not cer_values:
        return None
    return sum(cer_values) / len(cer_values)


# ── Harness runner ───────────────────────────────────────────────────────────


def evaluate_segmenter(
    heldout: list[dict[str, Any]],
    predict_fn: Any,
    *,
    recognizer_model_path: str | None = None,
    distance_tolerance: float = 15.0,
    reports_dir: str = "reports",
    label: str = "segmenter",
) -> dict[str, Any]:
    """Evaluate a segmenter on a frozen held-out fragment set.

    Args:
        heldout: list of ``{"image": Path, "gt_xml": Path}`` entries, one per
            held-out plate. The set is frozen (selected once, never trained on).
        predict_fn: callable ``(image_path: Path) -> dict`` returning the
            unified layout dict (``{"image": str, "lines": [...]}``). The
            caller wraps the chosen backend (blla / orli / YOLO) in this
            signature so the harness stays backend-agnostic.
        recognizer_model_path: optional path to a Kraken ``.safetensors``
            recognition model. When provided, the harness also computes
            end-to-end CER (metric 6). When ``None``, end-to-end CER is
            skipped and only layout metrics 1–5 are reported.
        distance_tolerance: baseline matching tolerance in pixels (default 15,
            matching the existing ``evaluate-layout`` CLI default).
        reports_dir: directory for the JSON report.
        label: label for the report filename (e.g. ``"blla"``, ``"orli"``,
            ``"yolo-obb"``).

    Returns:
        Aggregated report dict with per-plate and aggregate metrics. Writes
        ``reports/{label}__segmenter.json``.
    """
    from msocr.data.session_manager import SessionManager
    import tempfile

    per_plate: dict[str, dict[str, Any]] = {}
    aggregate_cer: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        manager = SessionManager(Path(tmp) / "sessions")
        for entry in heldout:
            image_path = Path(entry["image"])
            gt_xml = Path(entry["gt_xml"])
            plate_id = entry.get("id", image_path.stem)

            gt_state = manager.parse_page_xml_to_v2(gt_xml.read_bytes())
            if gt_state is None:
                logger.warning("Plate %s: GT XML unparseable; skipping", plate_id)
                continue

            gt_dict = gt_state_to_layout_dict(gt_state)
            layout = predict_fn(image_path)

            layout_result = evaluate_layout(
                gt_dict, layout, distance_tolerance=distance_tolerance,
            )

            plate_report: dict[str, Any] = dict(layout_result)

            if recognizer_model_path:
                e2e_cer = _end_to_end_cer(
                    image_path, layout, gt_state,
                    recognizer_model_path=recognizer_model_path,
                )
                plate_report["end_to_end_cer"] = e2e_cer
                if e2e_cer is not None:
                    aggregate_cer.append(e2e_cer)
            per_plate[plate_id] = plate_report

    # Aggregate: mean of per-plate metrics
    aggregate: dict[str, Any] = {}
    for metric in ("baseline_precision", "baseline_recall", "baseline_f1",
                   "mean_baseline_coverage", "reading_order_pair_accuracy"):
        vals = [p[metric] for p in per_plate.values()
                if p.get(metric) is not None]
        if vals:
            aggregate[metric] = sum(vals) / len(vals)
    for metric in ("split_count", "merge_count", "clipped_match_count"):
        vals = [p.get(metric, 0) for p in per_plate.values()]
        if vals:
            aggregate[metric] = sum(vals)
    if aggregate_cer:
        aggregate["end_to_end_cer"] = sum(aggregate_cer) / len(aggregate_cer)

    report = {
        "label": label,
        "plates_evaluated": len(per_plate),
        "distance_tolerance": distance_tolerance,
        "recognizer_model": recognizer_model_path,
        "per_plate": per_plate,
        "aggregate": aggregate,
    }

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / f"{label}__segmenter.json"
    out_path.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    return report


# ── Predict-function builders for each backend ──────────────────────────────
# These wrap a loaded model into the ``predict_fn(image_path) -> dict``
# signature the harness expects. The caller picks one and passes it to
# ``evaluate_segmenter``.


def make_kraken_predict_fn(model_path: str | None = None,
                           text_direction: str = "horizontal-rl"):
    """Build a predict_fn from a Kraken SegmentationTaskModel.

    ``model_path=None`` loads the default BLLA model (the zero-shot floor).
    Pass a path to load orli, D-FINE, or a fine-tuned BLLA via the Kraken 7
    plugin system.
    """
    from kraken.tasks import SegmentationTaskModel
    from kraken.configs import SegmentationInferenceConfig
    from PIL import Image

    if model_path:
        seg_model = SegmentationTaskModel.load_model(model_path)
    else:
        seg_model = SegmentationTaskModel.load_model()
    config = SegmentationInferenceConfig(text_direction=text_direction)

    def predict_fn(image_path: Path) -> dict[str, Any]:
        with Image.open(image_path) as img:
            seg = seg_model.predict(img, config)
        return kraken_segmentation_to_layout(seg, str(image_path))

    return predict_fn


def make_yolo_predict_fn(model_path: str):
    """Build a predict_fn from a YOLO-OBB model (Ultralytics)."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    def predict_fn(image_path: Path) -> dict[str, Any]:
        results = model(str(image_path))
        return yolo_obb_to_layout(results, str(image_path))

    return predict_fn


# ── Self-check ──────────────────────────────────────────────────────────────


def _self_check() -> None:
    """Smoke test the adapters + GT conversion with synthetic data."""
    # GT state shape from parse_page_xml_to_v2
    gt_state = {
        "lines": [
            {"id": "g1", "baseline": [(100, 10), (50, 10)],
             "boundary": [(100, 0), (50, 0), (50, 20), (100, 20)],
             "transcript": "abc", "rowId": "row-1"},
        ],
        "gaps": [],
    }
    gt_dict = gt_state_to_layout_dict(gt_state)
    assert gt_dict["lines"][0]["baseline"] == [[100, 10], [50, 10]]
    assert gt_dict["lines"][0]["polygon"] == [[100, 0], [50, 0], [50, 20], [100, 20]]

    # Unified layout shape from a Kraken-like seg (duck-typed)
    class _FakeLine:
        id = "p1"
        baseline = [(100, 11), (50, 11)]
        boundary = [(100, 1), (50, 1), (50, 21), (100, 21)]
    class _FakeSeg:
        lines = [_FakeLine()]
    layout = kraken_segmentation_to_layout(_FakeSeg(), "fake.png")
    assert layout["lines"][0]["id"] == "p1"
    assert layout["lines"][0]["bbox"] == [50, 1, 100, 21]

    # evaluate_layout should accept both
    result = evaluate_layout(gt_dict, layout, distance_tolerance=5)
    assert result["baseline_f1"] == 1.0, f"expected F1=1.0, got {result['baseline_f1']}"
    print("segmenter_harness self-check OK")


if __name__ == "__main__":
    _self_check()