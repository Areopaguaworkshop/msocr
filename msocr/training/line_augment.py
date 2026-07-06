"""Line-level image degradation for data augmentation.

Ported from kraken/linegen.py @ 9a218ce8 (mittagessen/kraken) which was removed
in kraken 7.x. Only the three degradation functions are kept; the Pango/Cairo
LineGenerator is intentionally dropped — we degrade real Sogdian line crops,
not synthesize from fonts.

Reference: Kanungo et al. "A statistical, nonparametric methodology for
document degradation model validation." IEEE TPAMI 22.11 (2000): 1209-1223.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import (
    affine_transform,
    binary_closing,
    distance_transform_cdt,
    find_objects,
    gaussian_filter,
    geometric_transform,
)

from kraken.lib.util import array2pil, pil2array

logger = logging.getLogger(__name__)

__all__ = ["ocropy_degrade", "degrade_line", "distort_line"]


# ponytail: verbatim port of kraken.linegen.ocropy_degrade (9a218ce8).
# Only logging stripped; behavior identical. Upgrade path: if scipy.ndimage
# filter/interpolation APIs move, re-port from upstream kraken tag.
def ocropy_degrade(
    im: Image.Image,
    distort: float = 1.0,
    dsigma: float = 20.0,
    eps: float = 0.03,
    delta: float = 0.3,
    degradations: tuple = ((0.5, 0.0, 0.5, 0.0),),
) -> Image.Image:
    """Degrade + distort a line using the ocropus noise model.

    Returns PIL.Image mode 'L'.
    """
    w, h = im.size
    image = Image.new("L", (int(1.5 * w), 4 * h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    a = pil2array(image.convert("L"))
    sigma, ssigma, threshold, sthreshold = degradations[np.random.choice(len(degradations))]
    sigma += (2 * np.random.rand() - 1) * ssigma
    threshold += (2 * np.random.rand() - 1) * sthreshold
    a = a * 1.0 / np.amax(a)
    if sigma > 0.0:
        a = gaussian_filter(a, sigma)
    a += np.clip(np.random.randn(*a.shape) * 0.2, -0.25, 0.25)
    m = np.array(
        [[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]]
    )
    w, h = a.shape
    c = np.array([w / 2.0, h / 2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    a = affine_transform(a, m, offset=d, order=1, mode="constant", cval=a[0, 0])
    a = np.array(a > threshold, "f")
    [[r, c]] = find_objects(np.array(a == 0, "i"))
    a = a[r.start - 5 : r.stop + 5, c.start - 5 : c.stop + 5]
    if distort > 0:
        h, w = a.shape
        hs = gaussian_filter(np.random.randn(h, w), dsigma)
        ws = gaussian_filter(np.random.randn(h, w), dsigma)
        hs *= distort / np.amax(hs)
        ws *= distort / np.amax(ws)

        def _f(p):
            return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

        a = geometric_transform(a, _f, output_shape=(h, w), order=1, mode="constant", cval=np.amax(a))
    return array2pil(a).convert("L")


# ponytail: verbatim port of kraken.linegen.degrade_line (9a218ce8).
def degrade_line(
    im: Image.Image,
    eta: float = 0.0,
    alpha: float = 1.5,
    beta: float = 1.5,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
) -> Image.Image:
    """Kanungo noise model: distance-weighted foreground/background flips.

    Returns PIL.Image mode 'L' (255 - binary*255).
    """
    im = pil2array(im)
    im = np.amax(im) - im
    im = im * 1.0 / np.amax(im)
    fg_dist = distance_transform_cdt(1 - im, metric="taxicab")
    fg_prob = alpha_0 * np.exp(-alpha * (fg_dist**2)) + eta
    fg_prob[im == 1] = 0
    fg_flip = np.random.binomial(1, fg_prob)
    bg_dist = distance_transform_cdt(im, metric="taxicab")
    bg_prob = beta_0 * np.exp(-beta * (bg_dist**2)) + eta
    bg_prob[im == 0] = 0
    bg_flip = np.random.binomial(1, bg_prob)
    im -= bg_flip
    im += fg_flip
    sel = np.array([[1, 1], [1, 1]])
    im = binary_closing(im, sel)
    return array2pil(255 - im.astype("B") * 255)


# ponytail: verbatim port of kraken.linegen.distort_line (9a218ce8).
# Run BEFORE degrade_line (adds 5px white border).
def distort_line(
    im: Image.Image,
    distort: float = 3.0,
    sigma: float = 10,
    eps: float = 0.03,
    delta: float = 0.3,
) -> Image.Image:
    """Geometric distortion: affine + smooth random displacement field.

    Returns PIL.Image mode 'L'.
    """
    w, h = im.size
    image = Image.new("L", (int(1.5 * w), 4 * h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    line = pil2array(image.convert("L"))
    m = np.array(
        [[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]]
    )
    c = np.array([w / 2.0, h / 2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    line = affine_transform(line, m, offset=d, order=1, mode="constant", cval=255)
    hs = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    ws = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    hs *= distort / np.amax(hs)
    ws *= distort / np.amax(ws)

    def _f(p):
        return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

    im = array2pil(geometric_transform(line, _f, order=1, mode="nearest"))
    im = im.crop(ImageOps.invert(im).getbbox())
    return im


if __name__ == "__main__":
    # Self-check: round-trip on a synthetic line, confirm all three ops run
    # and output is a valid L-mode image with finite pixel range.
    rng_line = Image.new("L", (100, 30), 255)
    import random

    for x in range(10, 90, 4):
        for y in range(8, 22):
            rng_line.putpixel((x, y), 0)
    np.random.seed(0)
    out_d = distort_line(rng_line)
    out_g = degrade_line(out_d)
    out_o = ocropy_degrade(out_g)
    assert out_o.mode == "L", out_o.mode
    arr = np.asarray(out_o)
    assert np.isfinite(arr).all(), "non-finite pixels"
    assert arr.min() >= 0 and arr.max() <= 255, (arr.min(), arr.max())
    print(f"line_augment self-check OK: distort→degrade→ocropy {out_o.size}")