"""Per-line damage confidence flag (Phase 0.5).

Cheap, no-pixel-annotation heuristic that flags lines a Kraken recognizer
likely failed to read because of damage or lacunae. Ships a usable
damage-triage signal immediately, while the pixel-level DamageZone
annotation campaign (see ``docs/DAMAGE_ANNOTATION_TASK.md``) runs in parallel.

A line is flagged ``likely_damage`` if ANY of:
- its transcript contains ``LACUNA_TOKEN`` (model learned to emit ░ for blanks),
- mean per-char confidence < ``LOW_CONF_THRESHOLD``,
- transcript is empty or whitespace-only,
- stripped transcript length is 1 (single grapheme, often a misread on
  damaged input).

ponytail: four OR'd thresholds, no ML. Ceiling: this flags ~any short or
low-conf line, not only damage — false positives expected. Upgrade path:
once DamageZone pixel labels exist (≥30 folios), train a dedicated damage
detector and replace this heuristic.
"""
from __future__ import annotations

from typing import Any

__all__ = ["flag_line", "flag_predictions", "LACUNA_TOKEN", "LOW_CONF_THRESHOLD"]

# Must match msocr.training.lacuna_augment.LACUNA_TOKEN
LACUNA_TOKEN = "░"
LOW_CONF_THRESHOLD = 0.3


def flag_line(record: dict[str, Any]) -> dict[str, Any]:
    """Add ``likely_damage`` (bool) and ``damage_reason`` (str | None) in place.

    ``record`` must have ``transcript`` (str) and ``confidence`` (float).
    Returns the same record dict for convenience.
    """
    transcript = record.get("transcript", "") or ""
    confidence = float(record.get("confidence", 0.0) or 0.0)
    stripped = transcript.strip()

    if LACUNA_TOKEN in transcript:
        record["likely_damage"] = True
        record["damage_reason"] = "lacuna_token"
    elif not stripped:
        record["likely_damage"] = True
        record["damage_reason"] = "empty_transcript"
    elif confidence < LOW_CONF_THRESHOLD:
        record["likely_damage"] = True
        record["damage_reason"] = "low_confidence"
    elif len(stripped) == 1:
        record["likely_damage"] = True
        record["damage_reason"] = "single_grapheme"
    else:
        record["likely_damage"] = False
        record["damage_reason"] = None
    return record


def flag_predictions(per_plate: list[dict]) -> list[dict]:
    """Flag every record in ``per_plate[i]['records']``. Mutates in place + returns."""
    for plate in per_plate:
        for rec in plate.get("records", []):
            flag_line(rec)
    return per_plate


# Integration: dump_preds.py can call flag_predictions(per_plate) after the
# predict loop.

if __name__ == "__main__":
    # Self-check: four records covering each branch.
    per_plate = [
        {
            "plate_id": "synth",
            "records": [
                {"line_id": "l1", "transcript": "ʾsʾn", "confidence": 0.92},  # clean
                {"line_id": "l2", "transcript": "???", "confidence": 0.18},   # low conf
                {"line_id": "l3", "transcript": "", "confidence": 0.5},       # empty
                {"line_id": "l4", "transcript": "░", "confidence": 0.6},      # lacuna
            ],
        }
    ]
    flag_predictions(per_plate)
    recs = per_plate[0]["records"]
    assert recs[0]["likely_damage"] is False, recs[0]
    assert recs[1]["likely_damage"] is True and recs[1]["damage_reason"] == "low_confidence", recs[1]
    assert recs[2]["likely_damage"] is True and recs[2]["damage_reason"] == "empty_transcript", recs[2]
    assert recs[3]["likely_damage"] is True and recs[3]["damage_reason"] == "lacuna_token", recs[3]
    print("damage_flag self-check OK")