import json

from PIL import Image

from msocr.segmentation.line_extraction import extract_lines_from_segments


def test_line_proposals_emit_padded_crops_and_review_provenance(tmp_path):
    pages = tmp_path / "pages"
    segments = tmp_path / "segments"
    lines = tmp_path / "lines"
    pages.mkdir()
    segments.mkdir()
    image_path = pages / "fragment.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    (segments / "fragment.segments.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "lines": [
                    {
                        "id": "proposal-1",
                        "boundary": [[30, 30], [170, 30], [170, 60], [30, 60]],
                        "baseline": [[35, 55], [165, 55]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance_path = tmp_path / "provenance.json"
    contact_path = tmp_path / "contact.jpg"

    count = extract_lines_from_segments(
        pages,
        segments,
        lines,
        provenance_path=provenance_path,
        contact_sheet_path=contact_path,
    )

    assert count == 1
    record = json.loads(provenance_path.read_text())["proposals"][0]
    assert record["bbox"][0] < 30
    assert record["bbox"][2] > 170
    assert record["training_eligible"] is False
    assert record["review_required"] is True
    assert contact_path.exists()
