# Damage annotation task — Phase 1 prerequisite

## Status: BLOCKED on annotation

A Phase 1 audit on {{ DATE }} found **zero DamageZone polygons** across all 14
annotated sessions (119 regions, all `MainZone`). The frontend, backend,
PAGE-XML export, and Kraken training path all support DamageZone — the only
missing input is the human annotation itself. This document specifies exactly
what to draw so Phase 1 can proceed.

## Goal

Produce **≥30 folios** with at least one `DamageZone` polygon each, so that:

- Phase 1 can hold out a clean damage-evaluation set.
- Kraken can be trained to *skip* damage regions during recognition.
- A lacuna-aware recognizer (Phase 0.5, built in parallel) can be validated
  against human-judged damage labels.

30 is the v3 threshold below which a held-out damage set is statistically
noise. More is better; fewer than ~20 is not worth training on.

## What counts as DamageZone

Draw a `DamageZone` polygon around any area of the folio where **the physical
manuscript is damaged or unreadable** and you would **not transcribe** the
text there (because there is no text, or it is illegible).

### Include

- **Lacunae** — holes, tears, missing corners where the parchment/paper is gone.
- **Abrasion / rubbing** — surface worn smooth, ink faded to nothing, text
  no longer readable.
- **Stains obscuring ink** — water damage, ink bleed, mud, wax, where the
  stain makes the underlying text unreadable. (Light stains that do not
  block reading: do **not** mark.)
- **Burn / char** — fire damage where text is destroyed.
- **Physical repairs overlapping text** — stitched patches, glued overlays
  that cover text you cannot read through them.
- **Mold / foxing** that has eaten the ink.

### Do NOT include

- **Digitization artefacts** — bleed-through from the verso, shadows,
  ruler marks, light glare, scanner dust. Use `DigitizationArtefactZone`
  instead. The rule: if it came from the scanner/camera, not the manuscript,
  it is not DamageZone.
- **Legible but faded text** — if you can still read it, transcribe it. Do
  not mark as damage. The model should learn faded-but-readable hands.
- **Marginalia or notes you do not want to transcribe** — use `MarginTextZone`
  or simply do not draw a baseline there. Damage is for *unreadable* areas,
  not *unwanted* areas.
- **Decorations / illustrations** — use `GraphicZone`.
- **Light, non-obscuring stains** — only mark stains that genuinely prevent
  reading.

### Edge case: partially damaged line

If a line is partly readable and partly destroyed:

1. Draw the baseline only over the **readable** portion.
2. Transcribe only the readable portion.
3. Draw a `DamageZone` polygon over the destroyed portion, **extending
   slightly past the line into the interlinear space above and below** so
   Kraken's polygonal environment does not try to read into it.

Do **not** try to transcribe a guess for destroyed text. Lacunae should be
blank in the transcript, not reconstructed with `[...]` or editorial marks.
Kraken does not learn from reconstructed text reliably.

## How to draw (in the annotation editor)

1. Press `R` (region mode).
2. In the top-left palette, click **DamageZone** (red, `#dc2626`).
3. Click points around the damaged area. Follow the **actual physical
   boundary** of the damage, not a tidy rectangle — Kraken uses the polygon
   to exclude pixels, and a tight boundary wastes fewer readable pixels.
4. Double-click to close.
5. You do **not** draw baselines inside a DamageZone. DamageZone is a
   recognition-exclusion region; there is nothing to read there.

### Tight vs. loose polygons

Prefer **tight** polygons. A loose polygon that eats into readable text
removes that text from training. A tight polygon that leaves a 2–3 px fringe
of damage around the edge is fine — the fringe is unreadable anyway. Aim for
the visible damage boundary ±2 px.

### One DamageZone per contiguous damaged area

If a folio has three separate lacunae, draw **three** DamageZone polygons.
Do not draw one big polygon spanning all three — that would exclude
readable text between them.

## Which folios to annotate

Priority order:

1. **The 14 existing c2av sessions** (`c2av01`–`c2av19` in
   `msocr/data/sessions/`). Re-open each, check whether the folio has any
   physical damage, and add DamageZone polygons where it does. Most c2av
   folios are clean — expect maybe 3–6 of the 14 to have any damage.
2. **New damaged folios** sourced from the c2av corpus or other Sogdian
   manuscript sources. Target: enough damaged folios to reach ≥30 total.
   This is the bulk of the work if the existing 14 are mostly clean.

### If the corpus does not have 30 damaged folios

Then Phase 1 cannot be fully unblocked and we rely on Phase 0.5 (lacuna-aware
recognizer, needs no pixel-level damage annotation) as the primary damage
signal. Report the count to the orchestrator so the plan can be re-scoped.

## Recording progress

After each folio, the editor auto-saves to `session.json` under
`annotations_v2.regions` with `type: "DamageZone"`. No separate log is
needed — the audit script (see below) reads all sessions and counts.

Run this to check progress anytime:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from collections import Counter
sd = Path("msocr/data/sessions")
dz = 0
folios_with_dz = 0
for s in sorted(sd.iterdir()):
    sj = s / "session.json"
    if not sj.exists(): continue
    v2 = json.loads(sj.read_text()).get("annotations_v2") or {}
    rc = Counter(r.get("type") for r in (v2.get("regions") or []))
    if rc.get("DamageZone", 0) > 0:
        folios_with_dz += 1
        dz += rc["DamageZone"]
print(f"Folios with >=1 DamageZone: {folios_with_dz}")
print(f"Total DamageZone polygons:  {dz}")
print(f"Target: 30 folios. Remaining: {max(0, 30 - folios_with_dz)}")
PY
```

## Done criteria

- [ ] ≥30 folios have ≥1 DamageZone polygon each.
- [ ] No DamageZone polygon covers legible text (spot-check 5 folios).
- [ ] No damaged area is marked as `MainZone` instead (spot-check: every
      folio with visible damage has a DamageZone, not just a MainZone
      over the whole column).
- [ ] Audit script reports `Folios with >=1 DamageZone: >= 30`.

When done, tell the orchestrator so Phase 1 engineering can proceed.