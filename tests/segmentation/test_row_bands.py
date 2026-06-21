from pathlib import Path

from PIL import Image, ImageDraw

from msocr.segmentation.row_bands import extract_row_bands


def test_extract_row_bands_keeps_fragmented_pieces_in_one_line(tmp_path: Path):
    image_path = tmp_path / "fragmented.png"
    image = Image.new("RGB", (260, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 55, 90):
        draw.rectangle((20, y, 70, y + 8), fill="black")
        draw.rectangle((170, y + 2, 230, y + 10), fill="black")
    image.save(image_path)

    bands = extract_row_bands(image_path, tmp_path / "out", expected_lines=3, min_component_area=10)

    assert [band.line_id for band in bands] == ["line_001", "line_002", "line_003"]
    assert len(list((tmp_path / "out" / "lines").glob("line_*.jpg"))) == 3
    assert all((band.bbox[2] - band.bbox[0]) > 180 for band in bands)
    assert (tmp_path / "out" / "line_overlay.jpg").exists()
    assert (tmp_path / "out" / "line_contact_sheet.jpg").exists()


def test_extract_row_bands_accepts_explicit_row_centers(tmp_path: Path):
    image_path = tmp_path / "fragmented.png"
    image = Image.new("RGB", (220, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 70, 28), fill="black")
    draw.rectangle((150, 22, 190, 30), fill="black")
    draw.rectangle((30, 60, 80, 68), fill="black")
    draw.rectangle((155, 62, 200, 70), fill="black")
    image.save(image_path)

    bands = extract_row_bands(
        image_path,
        tmp_path / "out",
        expected_lines=2,
        row_centers=[25, 65],
        min_component_area=10,
    )

    assert len(bands) == 2
    assert bands[0].bbox[1] < bands[1].bbox[1]
