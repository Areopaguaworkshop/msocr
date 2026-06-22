"""Binarization for geometry purposes (deskew, line-band clustering).

ponytail: NOT fed to BLLA — Kraken BLLA works on color/grayscale directly
(lib-4 confirmed binarization is deprecated for BLLA). These masks are used
only for Hough deskew angle detection and CC clustering.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola


def sauvola_binarize(image: Image.Image, window_size: int = 25) -> np.ndarray:
    """Sauvola adaptive thresholding. Returns binary mask (uint8, 0/255).

    Ink = 0 (black), background = 255 (white). Matches cv2 threshold convention.
    Handles uneven illumination better than Otsu on Turfan photos.
    """
    gray = np.array(image.convert("L"))
    thresh = threshold_sauvola(gray, window_size=window_size)
    # ink is darker than local threshold -> 0 (black); background -> 255 (white)
    binary = np.where(gray <= thresh, 0, 255).astype(np.uint8)
    return binary


def has_bleed_through(image: Image.Image) -> bool:
    """Detect bleed-through by checking for a third intermediate histogram mode.

    ponytail: clean high-contrast text is bimodal (ink + background, deep
    valley). Bleed-through adds faint text at an intermediate gray, producing
    a third mode between the ink and background peaks. If a third peak rises
    above 5% of the dominant peak, flag as bleed-through. Replace with an
    Otsu between-class variance ratio test if this misfires on real scans.
    """
    gray = np.array(image.convert("L"))
    hist = np.histogram(gray, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    # Smooth the histogram so flat regions don't produce spurious peaks.
    kernel = np.ones(5) / 5
    hist_s = np.convolve(hist, kernel, mode="same")
    max_peak = hist_s.max()
    if max_peak <= 0:
        return False
    # Local maxima: bin higher than its two neighbors.
    peaks = []
    for i in range(1, 255):
        if hist_s[i] > hist_s[i - 1] and hist_s[i] >= hist_s[i + 1]:
            peaks.append(i)
    # Keep peaks above 5% of the dominant peak.
    significant = [p for p in peaks if hist_s[p] >= 0.05 * max_peak]
    return len(significant) >= 3


def nlbin_binarize(image: Image.Image) -> np.ndarray:
    """Kraken nlbin binarization (deprecated but functional, lib-4 confirmed).

    Used only for fragments with bleed-through. Falls back to CLI subprocess
    if the Python import fails.
    """
    try:
        from kraken.binarization import nlbin

        bw = nlbin(image)
        return np.array(bw.convert("L"))
    except Exception:
        # ponytail: CLI subprocess fallback if Python import breaks in future Kraken
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_f:
            image.save(in_f.name)
            in_path = in_f.name
        out_path = in_path.replace(".png", "_bw.png")
        subprocess.run(["kraken", "-i", in_path, out_path, "binarize"], check=True)
        return np.array(Image.open(out_path).convert("L"))


def binarize_for_geometry(image: Image.Image, window_size: int = 25) -> np.ndarray:
    """Pick Sauvola or nlbin based on bleed-through detection.

    Default: Sauvola (fast, handles uneven illumination).
    If bleed-through detected: nlbin (handles bleed-through, slower).
    """
    if has_bleed_through(image):
        return nlbin_binarize(image)
    return sauvola_binarize(image, window_size=window_size)


if __name__ == "__main__":
    # Self-check: synthetic image with a dark text-like block on light bg
    img = Image.new("L", (200, 200), 240)
    for y in range(50, 150, 8):
        for x in range(30, 170):
            img.putpixel((x, y), 30)

    mask = sauvola_binarize(img, window_size=25)
    assert mask.shape == (200, 200), mask.shape
    assert mask.dtype == np.uint8
    # Text rows should be dark (0), background light (255)
    assert mask[50, 100] == 0, "text row should be black"
    assert mask[10, 10] == 255, "background should be white"

    # No bleed-through on this synthetic image
    assert has_bleed_through(img) is False, "clean image should not flag bleed-through"

    print("ok")