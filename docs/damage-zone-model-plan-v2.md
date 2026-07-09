# Fragmented Manuscript Damage-Zone Model — Research Verification & Revised Plan

> Companion to `fragmented-MS-model-training-plan.md` (2026-07-09). This document
> independently re-verifies the literature review, corrects one breaking tooling
> assumption, and adds one complementary approach the original plan didn't cover.
> Where this document disagrees with or extends the original, it says so explicitly.

**Status:** research verification + plan revision.
**Date:** 2026-07-09

---

## 1. TL;DR — what changes in your plan

1. **Phase 2's Kraken command is now wrong.** Kraken 7.0 removed `ketos segtrain
   --valid-regions` / `--merge-regions`. Class mapping moved to YAML experiment
   files. Section 4 below has the corrected invocation.
2. **Add a Phase 0.5: lacuna-aware `ketos train`.** A synthetic-lacuna fine-tune
   of your existing recognizer needs *zero* new pixel-level annotation, is
   cheaper than the mask gate, and gives you a second, independent damage
   signal (transcription log-probability) that composes with the vision-based
   mask later. This is new relative to your draft's three-approach framing —
   it's not a competitor to Approaches 1–3, it's a fourth, parallel track.
3. Everything else in your literature review (§2–3 of the original) checks out
   against independent search. A few numbers below sharpen your imbalance and
   effort estimates.

---

## 2. Verification of the original literature review

### 2.1 Confirmed as accurate

- **PLM-SegFormer** (Wang et al. 2024, *npj Heritage Science*): 70.1% mHit,
  51.2% mIoU, 10,064 folios in 12h — confirmed. One number worth adding: only
  **2.9% of pixels** in the full Sanskrit palm-leaf corpus are damage pixels,
  despite the corpus being described as seriously affected by damage overall.
  That's your imbalance benchmark — expect your Sogdian DamageZone class to be
  in a similar single-digit-percent range of total pixels, not just "rare in
  instance count." Source code and released weights are public
  (github.com/Ryan21wy/PLM_Segformer) — architecturally the closest published
  precedent to Approach 3, and plausibly usable as an initialization even
  across the palm-leaf → paper/parchment domain gap, since SegFormer's
  pretrained encoder is what's doing the heavy lifting, not palm-leaf-specific
  features.
- **YALTAi / Segmonto DamageZone**: confirmed 21 instances in the dataset as
  your draft states, but the more useful detail is that DamageZone and
  DigitizationArtefactZone are *over-represented in the test split* relative
  to training in the original paper's own data description. That's a signal
  the authors themselves found DamageZone too sparse to trust a train-set-only
  split for it — worth replicating that caution in your own eval design (§5
  below).
- **No Sogdian/Turfan-specific damage-detection CV work exists** — re-confirmed
  independently today. Also found incidental confirmation that some Berlin
  Turfan material has been imaged with phase-contrast X-ray radiology and
  hyperspectral imaging by the BBAW/MPIWG "Turfan Studies" project — not
  something you have access to, but worth knowing it exists in case a
  multispectral scan of your specific fragments ever surfaces; it would
  sidestep the RGB-only limitation entirely for those folios.
- **Kraken `segtrain` fine-tuning data thresholds** (30–50 pages book-specific,
  ~130 pages for ~41% from-scratch accuracy): confirmed against the Digital
  Orientalist and Digital Intellectuals sources your draft already cites.

### 2.2 New finds not in the original review

- **Indiscapes** (instance segmentation for Indic manuscripts, arXiv:1912.07025):
  explicitly reports that "Physical degradations" was their hardest-to-parse
  region class, attributing it to small footprint and inconsistent visual
  pattern — independent confirmation of the same failure mode YALTAi and PLM
  show, from a third, methodologically different system (Mask R-CNN-family
  instance segmentation rather than pixel classification or object detection).
- **MapSAM / MapSAM2** (adapting SAM/SAM2 to historical map imagery): a closer
  methodological analogue to your situation than PLM-SegFormer. It's
  foundation-model adaptation to a *non-photographic, degraded, historical*
  domain under label scarcity — same constraint shape as Sogdian folios (old,
  visually unlike SAM's natural-image training distribution, few labels
  available). If Approach 3 stalls on SegFormer/YOLO from-scratch fine-tuning,
  a SAM2-adapter route (LoRA-style adapter layers on a frozen SAM2 encoder) is
  a documented fallback with a lower annotation floor than full fine-tuning.
- **"Mind the Gap: Analyzing Lacunae with Transformer-Based Transcription"**
  (Borkar & Smith, ICDAR 2024 Workshop on Computational Paleography,
  arXiv:2407.00250) — this is the paper your draft's §3.2 table cites as
  "TrCro flagging lacuna lines w/o damage training." The model is TrOCR, not
  "TrCro" (likely a transcription slip in your notes). Precise numbers:
  - Adding synthetic-lacuna examples to TrOCR's supervised training raised
    lacuna-restoration success from **5.6% to 65.85%**.
  - Using the HTR model's own transcription **log-probability** as a feature
    in a logistic regression classifier flagged lines containing lacunae
    **53% of the time**, and lines with other transcription errors (ink
    problems, complex writing) **84% of the time** — without ever looking at
    the image.
  - Attention-weight-based flagging was tried and found *not* significantly
    better than log-probability alone — so you don't need to instrument
    attention maps, just track per-line confidence, which Kraken's
    `BaselineOCRRecord.confidences` already exposes per code point.

  This is the basis for the new Phase 0.5 below.

### 2.3 Kraken tooling — breaking change confirmed

Kraken 7.0 (current on PyPI as `kraken-7.0.2`, matching the `dfine_kraken`
plugin generation your other notes reference) made these changes relevant to
this project:

- `ketos segtrain --valid-regions <type>` and `--merge-regions <type>` are
  **removed**. Class filtering/merging is now defined in a YAML experiment
  file passed via `--config` before the subcommand.
- Training now produces Lightning `.ckpt` checkpoints as the primary artifact;
  the best checkpoint is converted to a weights file (default `safetensors`,
  not CoreML) at the end of a run. `--resume` continues an interrupted run
  from checkpoint state (checkpoint state is authoritative over CLI flags);
  `--load` starts a fresh run from existing weights.
- `--training-files`/`--evaluation-files` were renamed to
  `--training-data`/`--evaluation-data`.
- `ketos segtest` now scores against a configurable class mapping and reports
  baseline-detection metrics (Transkribus-style) instead of pixel
  accuracy/IoU only — relevant to how you'll define the "IoU ≥ 0.4" acceptance
  criterion in your draft's Phase 2 (§5): confirm which metric `segtest`
  reports on your installed version before committing to an IoU threshold in
  writing, since it may now hand you a baseline-detection F-score instead.

Corrected invocation is in §4.

---

## 3. Revised approach matrix

Your draft's three approaches (mask-at-inference, Kraken segtrain fine-tune,
dedicated SegFormer/YOLO model) stand. Adding a fourth:

| # | Approach | Needs pixel-level DamageZone labels? | Effort | GPU? | Output |
|---|---|---|---|---|---|
| 1 | Kraken `--mask` at inference | No (uses existing polygons) | Minimal | No | Mask gate on already-annotated folios |
| 2 | `ketos segtrain` fine-tune (YAML class mapping) | Yes | Medium | Yes | Auto-mask on new folios |
| 3 | Dedicated SegFormer/YOLOv8-seg (or SAM2-adapter) | Yes | Higher | Yes (transfer) | Best long-term mask; publishable |
| **4 (new)** | **Lacuna-aware `ketos train`** (synthetic blanking of existing line GT) | **No** | **Low** | Optional | Per-line confidence flag, complements 1–3 |

Approach 4 doesn't produce a spatial mask — it produces a *per-line signal*
telling you a line probably crosses damage, using only the transcription
ground truth you already have from your Sophro Mhiro fine-tune (148–179 lines
of PAGE-XML). It's not a substitute for the vision-based mask (it can't tell
you *where* in a line the gap is, only *that* the line is suspect), but it's
free to build in parallel and gives you a sanity check: any folio your Phase 0
mask marks undamaged but Approach 4 flags as low-confidence is worth a second
look, and vice versa.

---

## 4. Corrected roadmap

### Phase 0 — Mask gate (unchanged from your draft, still correct)

`kraken segment -bl -m mask.png` with mask rendered from existing DamageZone
polygons. No Kraken version issue here — the `segment -m` inference-time mask
flag is untouched in 7.0.

### Phase 0.5 — Lacuna-aware recognizer (new, do in parallel with Phase 0)

- Write a synthetic-lacuna augmentation pass over your existing line-image
  training set: randomly blank contiguous spans (mimicking hole/tear
  geometry — not random pixels) in a fraction of training lines, with the
  ground-truth transcript marking the span (Leiden-convention-style bracket
  token, matching Borkar & Smith's setup, adaptable to your Sogdian charset).
- Fine-tune your existing Sophro Mhiro-based Kraken recognizer on the
  augmented set via ordinary `ketos train` — no CLI changes needed here, this
  is recognition training, not segmentation training, so the 7.0
  `valid-regions` removal doesn't apply.
- At inference, log per-line mean confidence from
  `BaselineOCRRecord.confidences` (already computed) and flag lines below a
  threshold you calibrate against your held-out damage eval set (§5).
- Acceptance: flagged-line precision/recall against the same held-out
  damaged-folio set you'll build for Phase 2, benchmarked against the 53%/84%
  Borkar & Smith numbers as a sanity floor — you should beat their generic
  historical-document baseline given Sogdian's narrower domain.

### Phase 1 — Data consolidation (unchanged, still the bottleneck)

No change to your draft's plan here: export DamageZone polygons from
`session_manager.py` state into a mask/manifest format, decide subclass
granularity, build the held-out damage eval set. This remains the
prerequisite for both Phase 2 and Phase 4.

### Phase 2 — Kraken seg fine-tune (corrected CLI)

Your draft's command:
```
ketos segtrain --valid-regions DamageZone --valid-regions MainZone … --base blla.mlmodel
```
no longer works on Kraken 7.x. Corrected pattern — build a YAML experiment
file and pass it via `--config`:

```yaml
# damage_segtrain.yaml
precision: 32-true
device: auto
segtrain:
  training_data:
    - seg_train.lst
  evaluation_data:
    - seg_val.lst
  format_type: xml
  load: blla.mlmodel        # base model to fine-tune from
  checkpoint_path: seg_checkpoints
  weights_format: safetensors
  region_class_mapping:
    - ['DamageZone', 3]
    - ['MainZone', 4]
    # add remaining Segmonto/DamageZone-subclass types as needed,
    # or merge everything else into one background class
```

```
ketos --config damage_segtrain.yaml segtrain
```

Verify the exact YAML key names (`region_class_mapping` vs. `line_class_mapping`
vs. a combined key) against `kraken.re/main/training/segtrain.html` and
`ketos --help` on your installed 7.0.2 before running — the release notes
show `line_class_mapping` explicitly for line types; confirm the region
equivalent exists under the same or a sibling key, since the docs excerpt
available doesn't fully disambiguate the two. This is a five-minute check
that will save you a failed run.

Also re-verify your Phase 2 acceptance criterion (IoU ≥ 0.4) against what
`ketos segtest` actually reports on 7.0 — it may now emphasize baseline
detection F-scores over region IoU by default.

### Phase 3 — Dedicated damage model (unchanged, with one new fallback option)

Your draft's plan stands: SegFormer or YOLOv8-seg fine-tune, only if Phase 2
plateaus and ≥50 damaged folios are annotated. Add **SAM2 + adapter layers**
(MapSAM2-style) as a fallback if SegFormer from-scratch fine-tuning
underperforms at your data scale — it's designed for exactly the "foundation
model, historical/degraded domain, few labels" situation you're in, more so
than SegFormer's ImageNet-pretrained-encoder starting point.

---

## 5. Held-out eval set — one addition to your draft's §5/§8

Your draft correctly flags "how many held-out damaged folios, who transcribes
them" as an open question (§8.3). Given the YALTAi authors themselves had to
deliberately over-sample DamageZone into their test split rather than relying
on a random split, do the same here: don't randomly hold out folios and hope
enough have damage. Deliberately select your held-out set to guarantee damage
representation, and stratify by damage severity (light foxing vs. holes vs.
charring) if you're tracking subclasses at all, even if the training run
itself collapses to a single `DamageZone` class per your v1 recommendation.

---

## 6. Updated risk list

Everything in your draft's §7 stands. Two additions:

- **Kraken 7.0 API churn is ongoing.** The class-mapping migration happened
  recently enough that Digital Orientalist/Digital Intellectuals tutorials
  and most StackOverflow-adjacent Kraken guidance still show the old
  `--valid-regions` syntax. Budget time for docs/reality mismatches beyond
  just this one flag when you actually run Phase 2.
- **Approach 4's synthetic lacunae won't match real damage geometry.**
  Borkar & Smith's synthetic gaps are a simplification of real hole/tear/stain
  shapes. Treat Phase 0.5's confidence signal as a coarse triage tool, not a
  substitute for the vision-based mask — use it to prioritize which folios
  get human review first, not as ground truth for Phase 1's dataset.

---

## 7. Decision points (in addition to your draft's §8)

4. **Do you want Phase 0.5 built now, in parallel with Phase 1 data
   consolidation?** It reuses your existing HTR training pipeline
   (`ketos_trainer.py`) with no new annotation format, so it could plausibly
   ship before Phase 0's mask gate given the mask gate needs the DamageZone
   polygon-to-raster rendering step built first.
5. **Confirm the YAML key names** for `ketos segtrain` region class mapping
   on your installed Kraken 7.0.2 before Phase 2 implementation begins — a
   direct `ketos segtrain --help` / doc check, five minutes, avoids a wasted
   training run.

---

## References (new to this document)

- Borkar & Smith, "Mind the Gap: Analyzing Lacunae with Transformer-Based
  Transcription," ICDAR 2024 Workshop on Computational Paleography —
  [arXiv:2407.00250](https://arxiv.org/abs/2407.00250)
- Indiscapes: Instance Segmentation Networks for Layout Parsing of Historical
  Indic Manuscripts — [arXiv:1912.07025](https://arxiv.org/pdf/1912.07025)
- MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images —
  [arXiv:2510.27547](https://arxiv.org/pdf/2510.27547)
- MapSAM: Adapting Segment Anything Model for Automated Feature Detection in
  Historical Maps — [arXiv:2411.06971](https://arxiv.org/pdf/2411.06971)
- Kraken 7.0 release notes — [github.com/mittagessen/kraken/releases](https://github.com/mittagessen/kraken/releases)
- Kraken segmentation training docs (current) — [kraken.re/main/training/segtrain.html](https://kraken.re/main/training/segtrain.html)
- PLM-SegFormer / PLM_Segformer repo — [github.com/Ryan21wy/PLM_Segformer](https://github.com/Ryan21wy/PLM_Segformer)
- Turfan Studies (BBAW) — hyperspectral/X-ray imaging note —
  [mpiwg-berlin.mpg.de](https://www.mpiwg-berlin.mpg.de/feature-story/objects-revealed-unveiling-astral-manuscripts-berlin-turfan-collection)
