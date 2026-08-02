import xml.etree.ElementTree as ET

import pytest

from msocr.data.session_manager import encode_msocr_custom
from msocr.training.page_export import compile_training_page_xml

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _write_page(path, lines):
    root = ET.Element("PcGts", xmlns=NS)
    page = ET.SubElement(root, "Page", imageFilename="page.png")
    region = ET.SubElement(page, "TextRegion", id="r1")
    for line_id, transcript, metadata, baseline in lines:
        line = ET.SubElement(
            region,
            "TextLine",
            id=line_id,
            custom=encode_msocr_custom("structure {type:DefaultLine;}", metadata),
        )
        if baseline:
            ET.SubElement(line, "Baseline", points=baseline)
        text_equiv = ET.SubElement(line, "TextEquiv")
        ET.SubElement(text_equiv, "Unicode").text = transcript
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_compiler_physically_omits_excluded_lines(tmp_path):
    source = tmp_path / "full.xml"
    output = tmp_path / "training.xml"
    audit_path = tmp_path / "audit.json"
    _write_page(
        source,
        [
            ("keep", "abc", {"trainable": True}, "10,20 40,20"),
            (
                "drop",
                "def",
                {"trainable": False, "exclusionReason": "bisected edge glyph"},
                "50,20 80,20",
            ),
        ],
    )

    audit = compile_training_page_xml(source, output, audit_path=audit_path)

    root = ET.parse(output).getroot()
    ids = [line.get("id") for line in root.iter(f"{{{NS}}}TextLine")]
    assert ids == ["keep"]
    assert audit["included_count"] == 1
    assert audit["excluded"] == [{"line_id": "drop", "reason": "bisected edge glyph"}]
    assert audit_path.exists()


def test_compiler_rejects_presentation_gap_token(tmp_path):
    source = tmp_path / "full.xml"
    _write_page(
        source,
        [
            ("line", "abc [gap] def", {}, "10,20 80,20"),
        ],
    )

    with pytest.raises(ValueError, match="structural gap metadata"):
        compile_training_page_xml(source, tmp_path / "training.xml")


def test_compiler_omits_baseline_crossing_declared_gap(tmp_path):
    source = tmp_path / "full.xml"
    gap = {
        "id": "gap-1",
        "rowId": "row-1",
        "afterLineId": "crosses",
        "beforeLineId": "piece",
        "type": "hole",
        "polygon": [[40, 0], [60, 0], [60, 40], [40, 40]],
        "confidence": None,
    }
    _write_page(
        source,
        [
            ("crosses", "abc", {"gapAfter": gap}, "10,20 90,20"),
            ("piece", "def", {}, "65,30 90,30"),
        ],
    )

    audit = compile_training_page_xml(source, tmp_path / "training.xml")

    assert audit["included_count"] == 1
    assert audit["excluded"] == [
        {"line_id": "crosses", "reason": "baseline crosses declared destructive gap"}
    ]


def test_compiler_detects_narrow_gap_between_sparse_baseline_points(tmp_path):
    source = tmp_path / "full.xml"
    gap = {
        "id": "gap-1",
        "rowId": "row-1",
        "afterLineId": "crosses",
        "beforeLineId": "piece",
        "type": "hole",
        "polygon": [[491, 0], [494, 0], [494, 40], [491, 40]],
        "confidence": None,
    }
    _write_page(
        source,
        [
            ("crosses", "abc", {"gapAfter": gap}, "0,20 1000,20"),
            ("piece", "def", {}, "500,30 600,30"),
        ],
    )

    audit = compile_training_page_xml(source, tmp_path / "training.xml")

    assert audit["included_count"] == 1
    assert audit["excluded"] == [
        {"line_id": "crosses", "reason": "baseline crosses declared destructive gap"}
    ]
