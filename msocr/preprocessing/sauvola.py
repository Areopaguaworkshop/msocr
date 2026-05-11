"""Sauvola-based preprocessing for Payne-Smith pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from skimage import filters


def _deskew(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    if not angles:
        return image
    median_angle = np.median(angles)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_directory(input_dir: Path, output_dir: Path, cfg: Dict) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in input_dir.rglob("*.png"):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if cfg.get("deskew", True):
            img = _deskew(img)
        if cfg.get("denoise", True):
            img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        window = int(cfg.get("sauvola_window", 31))
        k = float(cfg.get("sauvola_k", 0.15))
        thresh = filters.threshold_sauvola(img, window_size=window, k=k)
        binary = (img > thresh).astype(np.uint8) * 255
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), binary)
        count += 1
    return count
