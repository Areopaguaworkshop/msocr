"""Detect the real manuscript text block on a page (margins/headers/footers ignored).

ponytail: union bbox of all ink components above a min area, padded. No density-
based filtering, no header/footer heuristics. Add those if marginal noise becomes
common — the union bbox is fooled by stray ink in margins.
"""

from __future__ import annotations

from PIL import Image

from msocr.segmentation.row_bands import _ink_components, _clamp_roi


def detect_manuscript_area(
    image: Image.Image, min_area: int = 50, margin_pad: int = 20
) -> tuple[int, int, int, int] | None:
    """Find the union bounding box of all ink components above ``min_area``.

    Returns ``(left, top, right, bottom)`` in original page pixels, clamped to
    the image bounds, or ``None`` if no ink components are found.
    """
    components = _ink_components(image, min_area)
    if not components:
        return None

    left = min(c[0] for c in components)
    top = min(c[1] for c in components)
    right = max(c[2] for c in components)
    bottom = max(c[3] for c in components)

    return _clamp_roi(
        (left - margin_pad, top - margin_pad, right + margin_pad, bottom + margin_pad),
        image.size,
    )


def crop_to_manuscript_area(
    image: Image.Image, roi: tuple[int, int, int, int] | None
) -> tuple[Image.Image, int, int]:
    """Crop ``image`` to ``roi``. Returns ``(cropped_image, off_x, off_y)``.

    If ``roi`` is None, returns the original image unchanged with offset (0, 0).
    """
    if roi is None:
        return image, 0, 0
    left, top, _, _ = roi
    return image.crop(roi), left, top


if __name__ == "__main__":
    # Self-check: synthetic page with a centered ink block on white background.
    page = Image.new("L", (400, 600), 255)
    # draw a dark rectangle roughly in the center (the "manuscript block")
    for y in range(180, 420):
        for x in range(120, 280):
            page.putpixel((x, y), 20)

    roi = detect_manuscript_area(page, min_area=50, margin_pad=20)
    assert roi is not None, "expected a detected roi for the ink block"
    left, top, right, bottom = roi
    # The detected box must contain the ink block (with pad slack).
    assert left <= 120 and top <= 180 and right >= 280 and bottom >= 420, roi
    # And must not be the whole page (margins should be trimmed).
    assert (right - left) < 400 and (bottom - top) < 600, roi

    cropped, off_x, off_y = crop_to_manuscript_area(page, roi)
    assert off_x == left and off_y == top, (off_x, off_y, left, top)
    assert cropped.size == (right - left, bottom - top), (cropped.size, roi)

    # No-components case: None roi -> passthrough.
    blank = Image.new("L", (100, 100), 255)
    assert detect_manuscript_area(blank) is None
    img2, ox, oy = crop_to_manuscript_area(blank, None)
    assert (img2.size, ox, oy) == ((100, 100), 0, 0)

    print("ok")