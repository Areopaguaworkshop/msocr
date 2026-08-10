"""Tests for msocr.evaluation.segmenter_harness (Phase 2 eval harness)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from msocr.evaluation.segmenter_harness import (
    evaluate_segmenter,
    gt_state_to_layout_dict,
    kraken_segmentation_to_layout,
    yolo_obb_to_layout,
)


# ── Adapter unit tests ──────────────────────────────────────────────────────


def test_gt_state_to_layout_dict_normalizes_tuples_to_lists():
    gt_state = {
        "lines": [
            {"id": "g1", "baseline": [(100, 10), (50, 10)],
             "boundary": [(100, 0), (50, 0), (50, 20), (100, 20)],
             "transcript": "abc", "rowId": "row-1"},
        ],
        "gaps": [{"polygon": [(42, 0), (48, 0), (48, 20), (42, 20)],
                  "afterLineId": "g1", "rowId": "row-1"}],
    }
    result = gt_state_to_layout_dict(gt_state)
    assert result["lines"][0]["baseline"] == [[100, 10], [50, 10]]
    assert result["lines"][0]["polygon"] == [[100, 0], [50, 0], [50, 20], [100, 20]]
    assert result["lines"][0]["rowId"] == "row-1"
    assert result["gaps"][0]["polygon"] == [[42, 0], [48, 0], [48, 20], [42, 20]]


def test_gt_state_to_layout_dict_handles_empty():
    result = gt_state_to_layout_dict({"lines": [], "gaps": []})
    assert result == {"lines": [], "gaps": []}


def test_kraken_segmentation_to_layout_extracts_baseline_boundary_bbox():
    class _FakeLine:
        id = "p1"
        baseline = [(100, 11), (50, 11)]
        boundary = [(100, 1), (50, 1), (50, 21), (100, 21)]
    class _FakeSeg:
        lines = [_FakeLine()]

    layout = kraken_segmentation_to_layout(_FakeSeg(), "fake.png")
    assert layout["image"] == "fake.png"
    assert layout["lines"][0]["id"] == "p1"
    assert layout["lines"][0]["baseline"] == [[100, 11], [50, 11]]
    assert layout["lines"][0]["polygon"] == [[100, 1], [50, 1], [50, 21], [100, 21]]
    assert layout["lines"][0]["bbox"] == [50, 1, 100, 21]


def test_kraken_segmentation_to_layout_handles_missing_boundary():
    class _FakeLine:
        id = "p2"
        baseline = [(10, 5)]
        boundary = None
    class _FakeSeg:
        lines = [_FakeLine()]

    layout = kraken_segmentation_to_layout(_FakeSeg(), "fake.png")
    assert layout["lines"][0]["polygon"] is None
    assert layout["lines"][0]["bbox"] is None


def test_yolo_obb_to_layout_converts_corners_to_baseline_polygon():
    import numpy as np

    class _FakeOBB:
        xyxyxyxy = np.array([[[10, 1], [60, 1], [60, 21], [10, 21]]])  # (1,4,2)
        conf = np.array([0.92])
        cls = np.array([0])
    class _FakeResult:
        obb = _FakeOBB()
        names = {0: "line"}
    results = [_FakeResult()]

    layout = yolo_obb_to_layout(results, "fake.png")
    assert len(layout["lines"]) == 1
    line = layout["lines"][0]
    # bottom edge RTL: corners[2]=(60,21) → corners[3]=(10,21)
    assert line["baseline"] == [[60.0, 21.0], [10.0, 21.0]]
    # polygon = 4 corners + closing point
    assert len(line["polygon"]) == 5
    assert line["bbox"] == [10.0, 1.0, 60.0, 21.0]
    assert line["confidence"] == pytest.approx(0.92)
    assert line["class"] == "line"


# ── Harness runner test ─────────────────────────────────────────────────────


def test_evaluate_segmenter_writes_report_and_aggregates(tmp_path):
    """End-to-end harness test with a mocked predict_fn and synthetic GT XML."""
    # Synthetic PAGE XML with one line
    page_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="plate.png" imageWidth="200" imageHeight="100">
    <TextRegion id="r1" custom="structure {type:MainZone;}">
      <Coords points="0,0 200,0 200,100 0,100"/>
      <TextLine id="g1" custom="structure {type:DefaultLine;}">
        <Coords points="100,0 50,0 50,20 100,20"/>
        <Baseline points="100,10 50,10"/>
        <TextEquiv><Unicode>abc</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>"""
    gt_xml = tmp_path / "plate.xml"
    gt_xml.write_text(page_xml, encoding="utf-8")
    image = tmp_path / "plate.png"
    image.write_bytes(b"")  # harness doesn't open it (predict_fn is mocked)

    heldout = [{"id": "plate-1", "image": str(image), "gt_xml": str(gt_xml)}]

    # Mock predict_fn: returns a prediction close to GT (within tolerance)
    def predict_fn(img_path):
        return {
            "image": str(img_path),
            "lines": [{
                "id": "p1",
                "baseline": [[100, 11], [50, 11]],
                "polygon": [[100, 1], [50, 1], [50, 21], [100, 21]],
            }],
        }

    report = evaluate_segmenter(
        heldout=heldout,
        predict_fn=predict_fn,
        recognizer_model_path=None,  # skip end-to-end CER (no Kraken in unit test)
        reports_dir=str(tmp_path / "reports"),
        label="test-seg",
    )

    assert report["label"] == "test-seg"
    assert report["plates_evaluated"] == 1
    assert report["per_plate"]["plate-1"]["baseline_f1"] == 1.0
    assert report["aggregate"]["baseline_f1"] == 1.0
    # Report file written
    report_path = tmp_path / "reports" / "test-seg__segmenter.json"
    assert report_path.exists()
    written = json.loads(report_path.read_text())
    assert written["label"] == "test-seg"


def test_evaluate_segmenter_skips_unparseable_gt(tmp_path, caplog):
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<not-page-xml/>", encoding="utf-8")
    image = tmp_path / "plate.png"
    image.write_bytes(b"")
    heldout = [{"id": "plate-x", "image": str(image), "gt_xml": str(bad_xml)}]

    report = evaluate_segmenter(
        heldout=heldout,
        predict_fn=lambda p: {"image": str(p), "lines": []},
        reports_dir=str(tmp_path / "reports"),
        label="empty",
    )
    assert report["plates_evaluated"] == 0