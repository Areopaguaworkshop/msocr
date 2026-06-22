"""Per-fragment Hough deskew for multi-fragment Turfan photos.

ponytail: extends the existing correct_skew logic in preprocessor.py to
operate per-fragment and handle RTL angle wrapping. The existing
ManuscriptPreprocessor.correct_skew is kept for the legacy whole-page flow;
this module is the per-fragment path for the new pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detect_skew_angle(
    binary_mask: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
    min_line_length: int = 100,
    max_line_gap: int = 10,
) -> float:
    """Detect dominant text line skew angle in degrees from a binary mask.

    Returns the angle to rotate by to make text lines horizontal.
    Handles RTL: wraps Hough line angles to [-90, 90] before taking the median
    so that baselines running right-to-left (angle near +/-180) don't cancel out
    baselines running left-to-right (angle near 0).

    Returns 0.0 if no Hough lines found.
    """
    edges = cv2.Canny(binary_mask, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Wrap to [-90, 90]: a line at 175deg is equivalent to -5deg
        angle = (angle + 90) % 180 - 90
        # Only keep near-horizontal lines (text lines are horizontal after deskew)
        # ponytail: +/-45deg window filters out vertical strokes and noise
        if -45 <= angle <= 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew_image(image: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate image by -angle_deg to correct skew. Uses BORDER_REPLICATE to
    avoid introducing black borders that would confuse downstream CC."""
    if abs(angle_deg) < 0.1:
        return image
    arr = np.array(image)
    height, width = arr.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    corrected = cv2.warpAffine(
        arr, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(corrected)


def deskew_fragment(
    color_image: Image.Image, binary_mask: np.ndarray
) -> tuple[Image.Image, float]:
    """Full per-fragment deskew: detect angle from mask, rotate color image.

    Returns (deskewed_color_image, angle_detected_deg).
    The mask is NOT rotated back -- it was only used for angle detection.
    """
    angle = detect_skew_angle(binary_mask)
    deskewed = deskew_image(color_image, angle)
    return deskewed, angle


if __name__ == "__main__":
    # Self-check: synthetic skewed text block
    img = Image.new("RGB", (400, 300), (240, 240, 240))
    arr = np.array(img)
    for y in range(80, 220, 12):
        for x in range(50, 350):
            arr[y, x] = (30, 30, 30)
    img = Image.fromarray(arr)
    img_skewed = img.rotate(5.0, expand=False, fillcolor=(240, 240, 240), resample=Image.BICUBIC)
    gray = np.array(img_skewed.convert("L"))
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    angle = detect_skew_angle(mask)
    # Detected angle should be close to -5 (the rotation we applied)
    # ponytail: Hough is approximate; +/-2deg tolerance
    assert abs(angle - (-5.0)) < 2.5, f"detected angle {angle} should be near -5, got {angle}"
    deskewed, detected = deskew_fragment(img_skewed, mask)
    assert deskewed.size == img_skewed.size, "deskewed should preserve size"
    assert detected == angle, "deskew_fragment should return the detected angle"
    # No-lines case: blank mask
    blank_mask = np.full((100, 100), 255, dtype=np.uint8)
    assert detect_skew_angle(blank_mask) == 0.0, "blank mask should give 0 angle"
    # Tiny angle case: should return image unchanged
    tiny_img = Image.new("RGB", (50, 50), (200, 200, 200))
    out = deskew_image(tiny_img, 0.05)
    assert out.size == tiny_img.size, "tiny angle should still return same size"
    print("ok")