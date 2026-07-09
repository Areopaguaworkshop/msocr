"""Lacuna-blank line crop for synthetic damage augmentation (Phase 0.5).

Produces a fully-white line crop — the lacuna signal. Used by
``scripts/augment_lacuna.py`` to synthesize damaged training plates where a
random subset of lines is blanked and their transcript replaced with
``LACUNA_TOKEN``, so the recognizer learns to emit that token (with low
confidence) when it sees blank/damaged input.

ponytail: full blank is the simplest honest lacuna signal. A partial-blank
(blank_ratio: blank only a fraction of the crop width) is a YAGNI upgrade —
not implemented now. Upgrade path: add a ``blank_ratio`` kwarg when the full
blank proves too easy for the model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["lacuna_blank_line", "LACUNA_TOKEN"]

# U+2591 LIGHT SHADE — visible, no collision with Sogdian U+10F30–U+10F44,
# and kraken's codec treats it as a regular grapheme.
LACUNA_TOKEN = "░"


def lacuna_blank_line(line_crop: Image.Image) -> Image.Image:
    """Return a fully white (blank) version of ``line_crop``.

    Same size as input, mode ``'L'``, all pixels 255. The input is not
    inspected — blanking is unconditional.
    """
    w, h = line_crop.size
    return Image.new("L", (w, h), 255)


if __name__ == "__main__":
    # Self-check: synthesize a line with ink, blank it, assert all-white + size.
    rng_line = Image.new("L", (100, 30), 255)
    for x in range(10, 90, 4):
        for y in range(8, 22):
            rng_line.putpixel((x, y), 0)
    out = lacuna_blank_line(rng_line)
    assert out.mode == "L", out.mode
    assert out.size == rng_line.size, (out.size, rng_line.size)
    arr = np.asarray(out)
    assert arr.min() == 255 and arr.max() == 255, (arr.min(), arr.max())
    print("lacuna_augment self-check OK")