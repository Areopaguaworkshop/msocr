# Fragmented-Manuscript Line-Level Segmentation Model Plan — v1

**Date:** 2026-08-10
**Version:** v1
**Status:** Planning baseline; no implementation in this document
**Scope:** Adding a learned line-level segmentation model for fragmented Christian Sogdian (C2/C2AV / E27) manuscripts, to replace the current classical-CV `row_bands` workaround for end-to-end automation. Recognition training is out of scope for this plan.

Companion to:
- [Christian Sogdian Fragment HTR and CER-Reduction Plan](./2026-08-02-christian-sogdian-fragment-htr-plan-v1.md)
- [Kraken Fragmented-Manuscript Limitations and C2 Image-Source Report](./2026-08-02-kraken-fragmented-manuscript-limitations-v1.md)
- [Orli, Fragment-Aware Layout Training, and Kraken Maintainer Contact](./2026-08-10-orli-and-fragment-layout-training-research-v1.md)

---

## 0. Pipeline overview

The plan introduces two pipelines: a **pre-training pipeline** that produces
two models from one annotation corpus, and a **post-training pipeline** that
uses those two models for end-to-end automation with no human in the loop.

### 0.1 Pre-training pipeline (produces the two models)

```
                    C2AV publication plate
                            │
                            ▼
                ┌───────────────────────┐
                │  isolate → binarize   │
                │  → deskew             │  classical CV (Phase 1-3, unchanged)
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │  row_bands            │  classical CV bootstrap
                │  (human expected_lines)│  (no learned model yet)
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │  annotation SPA       │  human marks:
                │  (React + OpenSeadragon)│  baseline + polygon + transcript
                └───────────┬───────────┘  + row_id + RTL order + gap
                            ▼
                     ┌─────────────┐
                     │   PAGE XML  │  ONE annotation session
                     │  per plate  │  (the shared input for both models)
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │  line crops +    │        │  page image +        │
    │  transcripts     │        │  baseline polygons   │
    │  (from XML)      │        │  (from XML)          │
    └────────┬─────────┘        └──────────┬───────────┘
             ▼                             ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │  ketos train     │        │  orli fine-tune      │
    │  (RunPod GPU)    │        │  OR YOLOv11-OBB      │
    │                  │        │  OR D-FINE kraken    │
    │  recognition     │        │  (RunPod GPU)        │
    │  CTC loss        │        │  segmentation        │
    └────────┬─────────┘        └──────────┬───────────┘
             ▼                             ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │  recognizer      │        │  segmenter           │
    │  .safetensors    │        │  .safetensors / .pt  │
    │  MODEL 1         │        │  MODEL 2             │
    └──────────────────┘        └──────────────────────┘
```

### 0.2 Post-training pipeline (uses the two models, no human in the loop)

```
              new unseen C2AV plate
                      │
                      ▼
          ┌───────────────────────┐
          │  isolate → binarize   │
          │  → deskew             │  classical CV (unchanged)
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  SEGMENTER MODEL      │  ← MODEL 2 (learned)
          │  (orli / YOLO / D-FINE)│
          │  input: deskewed      │
          │         fragment image│
          │  output: per-line     │
          │         geometry      │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  layout.json          │  per-line:
          │  {                    │    line_id
          │    lines: [           │    baseline / bbox
          │      {line_id: L001,  │    polygon
          │       baseline: [...],│    (NO text yet)
          │       polygon: [...]},│
          │      ...              │
          │    ]                  │
          │  }                    │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  extract line crops   │  mechanical:
          │  from image using     │  crop image by polygon
          │  layout.json geometry │  per line_id
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  RECOGNIZER MODEL     │  ← MODEL 1 (Kraken)
          │  (Kraken .safetensors)│
          │  input: line crop     │
          │  output: text + conf  │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  ocr_text.json        │  per-line:
          │  {                    │    line_id  (from layout.json)
          │    lines: [           │    text     (from recognizer)
          │      {line_id: L001,  │    confidence
          │       text: "...",    │
          │       conf: 0.92},    │
          │      ...              │
          │    ]                  │
          │  }                    │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  msocr JOIN           │  ← uses human-confirmed metadata
          │  (the philology step) │     (row_id, RTL order, gaps)
          │                       │
          │  match by line_id:    │
          │    layout.json row_id │
          │    + ocr_text.json    │
          │                       │
          │  reassemble broken    │
          │  rows:                │
          │    L001 + L002 → row 3│
          │    insert [gap] at    │
          │    hole location      │
          │                       │
          │  order by RTL reading │
          │  order within row     │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  final output         │
          │  .json / .md / .tex   │
          │                       │
          │  structured text with │
          │  explicit [gap]       │
          │  markers + row        │
          │  metadata             │
          └───────────────────────┘
```

### 0.3 Bootstrap loop (how pre-training feeds post-training)

The segmenter is bootstrapped from classical CV (`row_bands`), then improves
annotation speed, which produces more training data, which retrains a better
segmenter:

```
 ┌─── Phase 1 (bootstrap) ───────────────────┐
 │                                           │
 │  plate → isolate → deskew                 │
 │     → row_bands (classical CV)            │
 │     → annotation SPA                      │
 │     → PAGE XML                            │
 │                                           │
 └───────────────────────────────────────────┘
                      │
                      ▼
 ┌─── Phase 3-5 (learned) ───────────────────┐
 │                                           │
 │  PAGE XML → train segmenter (MODEL 2)    │
 │  PAGE XML → train recognizer (MODEL 1)   │
 │                                           │
 │  THEN: plate → isolate → deskew           │
 │    → segmenter (learned) ← NEW            │
 │    → recognizer                           │
 │    → join → output                        │
 │                                           │
 │  AND: segmenter suggestions →             │
 │    annotation SPA (faster than            │
 │    drawing every line by hand)            │
 │    → more PAGE XML → retrain              │
 │                                           │
 └───────────────────────────────────────────┘
```

The annotation speedup arrow is the virtuous loop: once a trained segmenter
exists, it suggests baselines to the annotator, who corrects instead of
draws from scratch, which produces more PAGE XML faster, which retrains a
better segmenter.

---

## 1. Executive decision

The project today has **no learned segmenter** for fragments. Every
`SegmentationTaskModel.load_model()` call in the codebase uses Kraken's stock
`blla` model with no path argument, and `KetosTrainer` only wraps `ketos train`
+ `ketos test` (recognition). The Phase 1→2→3 pipeline + `extract_row_bands`
produces line crops via classical CV (Sauvola + CC + DBSCAN + k-means
Y-clustering with a human-supplied line count), not a trained model.

This is correct for producing **recognition training data** (the annotator
sees line crops; the recognizer never sees a fragment photo). It is
**insufficient for end-to-end automation** from a raw fragment photo, which
requires a learned segmenter to predict per-line geometry without a human in
the loop.

The plan therefore is: **add a learned line-level segmentation model for
fragments, trained from the same PAGE XML annotation the project already
produces, chosen from one of four candidate architectures, integrated behind
the existing `runtime` / `annotation_api` / `extract-lines` surface so the
classical-CV path remains a fallback.**

This plan does not pick the architecture now. It defines the decision
criteria, the data prerequisites that are common to all architectures, the
evaluation harness that will decide between them, and the phased rollout that
keeps the existing recognizer untouched.

---

## 2. Why a learned segmenter is needed (and what it does not replace)

### 2.1 What the current pipeline cannot do

| Capability | Current state | Needs a learned segmenter? |
|---|---|---|
| Produce recognition training data from a fragment photo | Yes — `row_bands` + human `expected_lines` | No |
| Run end-to-end HTR on a new unseen fragment photo with no human | **No** — `extract-lines` requires a human to count rows | **Yes** |
| Suggest line geometry to the annotator faster than the human can draw it | Partial — BLLA tried, failed (E27 pilot, zero usable crops) | **Yes** (a better suggestion engine) |
| Predict reading order on a fragment with split rows | No — `row_id` + RTL fragment order are human-confirmed metadata | Partial — orli predicts order natively; YOLO/D-FINE do not |
| Scale annotation to the remaining 87 of 99 C2AV plates | Slow — every plate needs manual row counting and boundary drawing | **Yes** (even a mediocre learned segmenter accelerates annotation 2–5×) |

### 2.2 What a learned segmenter does NOT replace

- **The recognizer.** `ketos train` continues unchanged. The recognizer
  consumes line crops; it never sees the fragment photo. A new segmenter
  changes what line crops get produced, not how text is recognized.
- **The `row_bands` pipeline.** It stays as a deterministic fallback for
  plates where the learned segmenter fails or for annotation bootstrap. The
  learned segmenter is an *addition*, not a replacement.
- **Human-confirmed `row_id` and gap metadata.** No architecture in scope
  models the philological relationship between disconnected pieces of one
  manuscript row. That stays msocr metadata per the locked decision in
  `2026-08-02-christian-sogdian-fragment-htr-plan-v1.md` §2.
- **The Phase 1→2→3 fragment isolation pipeline.** A learned segmenter
  consumes the *output* of Phase 1 (isolated, deskewed fragment image), not
  the raw publication plate. Isolation stays classical CV.

### 2.3 The precise scope boundary

```
publication plate
    │
    ▼
[OUT OF SCOPE — stays classical CV]
Phase 1 isolate-fragments → Phase 2 binarize → Phase 3 deskew
    │
    ▼ isolated, deskewed fragment image
    │
[IN SCOPE — this plan]
learned line-level segmenter  →  per-line geometry (baseline+polygon OR bbox)
    │
    ▼
[OUT OF SCOPE — stays Kraken recognition]
line-level OCR  →  per-line text
    │
    ▼
[OUT OF SCOPE — stays msocr metadata]
join with row_id + RTL fragment order + gap metadata → structured output
```

---

## 3. Candidate architectures (the decision space)

All four consume the same PAGE XML annotation (baselines + polygons +
transcripts). They differ in what they predict, how they're trained, and how
well they fit fragmented inputs.

| | **orli (fine-tune)** | **Kraken `segtrain` (BLLA fine-tune)** | **YOLOv11-OBB** | **D-FINE kraken** |
|---|---|---|---|---|
| **Paradigm** | Autoregressive image → baseline sequence | U-Net pixel heatmap → polygonize | Object detection (oriented bbox) | Object detection (axis-aligned bbox) |
| **Predicts** | Baseline curves + reading order | Baselines + regions (unordered) | Oriented bboxes (unordered) | Bboxes + region class (unordered) |
| **Input** | 1920×1440, bf16 only | Any size, any precision | ~640–1280, any precision | ~640–1280, any precision |
| **Annotation conversion** | None (PAGE XML native) | None (PAGE XML native) | Polygon → 4 OBB corners (~10 LoC) | Polygon → bbox (~5 LoC) |
| **Line curvature preserved?** | Yes (chord-frame polyline) | Yes (vectorized polyline) | No (rotated rectangle) | No (axis-aligned rectangle) |
| **Reading order native?** | **Yes** | No (separate heuristic) | No | No |
| **Fragment-fit (theoretical)** | Medium — autoregressive may handle sparse lines, but trained on full pages | Low — U-Net heatmap is the wrong inductive bias for sparse inputs (E27 pilot: zero usable crops) | **High** — instance detection is inherently sparse-input friendly | High — same as YOLO |
| **Fragment-fit (evidence)** | None published | None published; DSS project reports success but on larger fragments (5–6 lines avg) | YALTAi: YOLO beats Kraken on small datasets; YOLOv11-OBB SOTA on medieval MSS | None fragment-specific |
| **Kraken integration** | Kraken 7 plugin (entry-point) | Native (`ketos segtrain`) | External — needs custom bridge to feed bboxes as `Segmentation` | Kraken 7 plugin (entry-point) |
| **Recognition pipeline** | Direct — orli outputs baselines, Kraken recognizer consumes them | Direct — Kraken native | Indirect — bbox → baseline conversion or `--linetype bbox` (Kraken 7.0.3) | Indirect — same as YOLO |
| **Training data floor** | Unknown; paper fine-tunes on small sets with strong results | 30–50 pages (per Digital Orientalist tutorial) | YALTAi: ≤1110 samples competitive | LADaS-scale for pretraining; fine-tune data unknown |
| **Maturity** | Released June 2026, 103 commits | Mature, Kraken-native | Mature ecosystem (Ultralytics) | Released 2025, Kraken plugin |
| **Maintainer input** | **Unknown — must ask Kiessling** | Known (he built it) | Third-party | Built by Kiessling |

### 3.1 Honest ranking for the fragmented-Sogdian case

1. **YOLOv11-OBB** — best *theoretical* fit (instance detection is
   sparse-input friendly by construction), most mature training ecosystem,
   strong evidence on small datasets and medieval manuscripts. Loses line
   curvature; needs a bbox→baseline bridge or Kraken 7.0.3 `--linetype bbox`.
2. **orli (fine-tune)** — best *preservation* fit (keeps curved baselines
   and reading order), Kraken-native integration, paper shows strong OOD
   fine-tuning. Unknown fragment behavior; must ask the maintainer before
   committing.
3. **D-FINE kraken** — same detection paradigm as YOLO, already a Kraken
   plugin, built by the same maintainer. Less mature ecosystem than
   Ultralytics; loses curvature; loses reading order.
4. **Kraken `segtrain` (BLLA fine-tune)** — lowest risk to try (Kraken
   native, no new deps), but the U-Net heatmap is the *wrong* inductive bias
   for sparse fragments (the E27 pilot already proved the base model fails;
   fine-tuning a wrong-prior model on 12 plates is unlikely to flip that).

### 3.2 Why this plan does not pick one now

- **orli fragment behavior is unknown.** Emailing Kiessling
  (`mittagessen@l.unchti.me`) before investing in any fine-tuning is
  cheaper than a wasted RunPod run. The draft is in
  `2026-08-10-orli-and-fragment-layout-training-research-v1.md` §3.3.
- **The annotation corpus is too small to train any of them well.** 12
  annotated plates is below the floor for every architecture in the table.
  The first phase of this plan is annotation scaling, which is
  architecture-independent.
- **The evaluation harness doesn't exist yet.** Picking an architecture
  before you can measure it on a held-out fragment set is premature. The
  harness (Phase 2 below) is also architecture-independent.

---

## 4. Locked decisions

1. **A learned segmenter is needed** for end-to-end automation and
   annotation scaling. The classical-CV `row_bands` path stays as fallback
   and bootstrap; it is not the production future.
2. **The segmenter is a separate model from the recognizer.** Kraken's
   architecture forces this; the plan embraces it. The recognizer training
   pipeline (`ketos train` via `KetosTrainer` / `walk_style_group`) is
   untouched.
3. **One annotation session trains all candidate architectures.** The
   annotation format is PAGE XML (baseline + polygon + transcript), the
   richest common format. Conversions to OBB / bbox are deterministic and
   downstream. Annotating in a poorer format would close architectural
   options.
4. **The segmenter consumes isolated, deskewed fragment images**, not raw
   publication plates. Phase 1→2→3 isolation stays classical CV and
   upstream of the segmenter.
5. **The segmenter does not model `row_id` or gaps.** Those stay
   human-confirmed msocr metadata per
   `2026-08-02-christian-sogdian-fragment-htr-plan-v1.md` §2. The segmenter
   outputs per-line geometry; msocr joins lines into rows.
6. **Architecture selection happens after Phase 2 (evaluation harness)
   and the Kiessling email, not before.** Committing to orli or YOLO now
   would be premature given the unknowns.
5. **No code is written until this plan is reviewed and a phase is
   approved.** This document is a plan, not an implementation spec.

---

## 5. Phased delivery plan

Five phases. Phase 1 is annotation scaling (architecture-independent,
unblocks everything). Phase 2 is the evaluation harness
(architecture-independent, decides between candidates). Phases 3–5 are
architecture-specific and gated on Phase 2 results + the Kiessling email.

### Phase 0 — Maintainer outreach (days, no code)

**Goal:** de-risk the orli branch before committing any GPU time.

- Send the email draft in
  `2026-08-10-orli-and-fragment-layout-training-research-v1.md` §3.3 to
  `mittagessen@l.unchti.me` (cc `benjamin.kiessling@psl.eu` as fallback).
- Two questions only: (a) has orli been tested on fragmentary / non-page
  inputs? (b) does he have a view on orli vs bbox object detectors (D-FINE /
  YOLO-OBB) for fragment line detection, before we invest in fine-tuning
  either?
- If he responds, attach his answer to this plan as an addendum and update
  the Phase 3/4 ranking.
- If no response in 10 business days, proceed with the ranking in §3.1
  (YOLO-OBB first, orli second).

**Exit criteria:** email sent; response logged or 10-day deadline passed.

### Phase 1 — Annotation scaling (weeks, architecture-independent)

**Goal:** produce enough annotated fragment PAGE XML to train and evaluate
any candidate. This is the single highest-leverage phase; 12 plates is
below every architecture's floor.

- Annotate the remaining 87 C2AV plates using the existing React annotation
  SPA + `row_bands` bootstrap. Target: **≥40 plates annotated** before
  Phase 3 starts (a pragmatic floor; 80+ is better).
- Per-plate annotation must include, at minimum:
  - one baseline + one polygon per contiguous readable line fragment
  - transcript of visible ink (Leiden underdot for uncertain readings;
    exclude unidentifiable glyphs)
  - `row_id` + RTL fragment order + gap metadata (per the existing plan §2)
- No new annotation UI work. The existing SPA already produces this.
- Annotation export stays PAGE XML. No new export format.
- Track annotation throughput: plates/week, lines/plate, minutes/line. If
  throughput is too low to hit 40 plates in a reasonable window, stop and
  reassess — do not lower the floor.

**Exit criteria:** ≥40 plates annotated with full PAGE XML; corpus stats
recorded (plate count, line count, char count, fragment-count distribution,
line-length distribution).

### Phase 2 — Evaluation harness (days, architecture-independent)

**Goal:** define and build the measurement that will decide between
architectures. This extends the existing `evaluate-layout` command, not a
new system.

- Define a **held-out fragment evaluation set**: 6–10 annotated plates not
  used in any training split, chosen to span the difficulty range (small
  fragments, large fragments, holes, split rows, faint ink). Selected once,
  frozen.
- Define metrics, all already partially implemented in
  `evaluation/layout_metrics.py`:
  - **Baseline precision / recall / F1** within a distance tolerance
    (existing).
  - **Split count** — how many GT lines get multiple predicted lines
    (existing; the BLLA failure mode).
  - **Merge count** — how many predicted lines cover multiple GT lines
    (existing).
  - **Row-grouping accuracy** — does the predicted geometry preserve
    `row_id` grouping? (existing).
  - **Gap localization IoU** — does the predicted geometry avoid
    drawing baselines across holes? (existing).
  - **End-to-end CER** — feed predicted geometry into the existing
    recognizer, measure CER on visible transcribed characters. This is the
    metric that actually matters; it closes the loop from segmenter to
    recognizer to text.
- Build the harness as a thin wrapper over the existing `evaluate-layout`
  + `evaluate` commands. No new metrics invented; reuse what's there.
- Run the held-out set through the **stock blla baseline** first, as the
  zero-shot floor. The E27 pilot already predicted zero usable crops;
  confirm this on the larger held-out set and record the numbers.

**Exit criteria:** held-out set frozen; harness produces a single JSON
report per (architecture, model) pair with all six metrics; stock-blla
baseline numbers recorded as the floor to beat.

### Phase 3 — Candidate 1 fine-tune + evaluate (weeks)

**Goal:** train and evaluate the first candidate architecture on the
annotated corpus. Pick the candidate based on Phase 0 + §3.1 ranking.

- Default candidate if Kiessling does not respond or endorses the detection
  paradigm: **YOLOv11-OBB**. Reasons: best theoretical fragment-fit, most
  mature training ecosystem, strong small-dataset evidence (YALTAi).
- Default candidate if Kiessling endorses orli on fragments: **orli
  fine-tune**. Reasons: keeps curved baselines and reading order, Kraken
  plugin integration, no bbox→baseline bridge needed.
- Training:
  - Convert PAGE XML → candidate's training format (OBB txt for YOLO;
    native PAGE XML for orli). Conversion is deterministic, ~10–50 LoC.
  - Train/fine-tune on the Phase 1 corpus's train split, validate on the
    val split. RunPod GPU Pod (existing `runpod_runner.py` reused; no new
    infra).
  - Record training config, hyperparameters, wall-clock, cost.
- Evaluate on the Phase 2 held-out set with the Phase 2 harness. Compare
  against the stock-blla floor and against `row_bands` (classical-CV
  baseline, no training).
- **Decision gate:** does the candidate beat stock blla by a clear margin
  on end-to-end CER AND on split-count? If yes, proceed to Phase 5
  integration. If no, proceed to Phase 4 (try the other candidate).

**Exit criteria:** one trained candidate model; one Phase 2 harness report;
a documented yes/no on the decision gate.

### Phase 4 — Candidate 2 fine-tune + evaluate (weeks, conditional)

**Goal:** only run if Phase 3's candidate did not clear the decision gate,
or if Phase 3's margin is small enough that a second candidate is worth the
cost.

- Train the second-ranked candidate from §3.1 (orli if Phase 3 was YOLO;
  YOLO if Phase 3 was orli).
- Same harness, same held-out set, same metrics.
- Record a head-to-head comparison table.

**Exit criteria:** second trained model; head-to-head report; one
architecture selected for integration.

### Phase 5 — Integration (weeks)

**Goal:** wire the selected segmenter into the existing msocr runtime so
end-to-end automation works, without removing the classical-CV fallback.

- Add a `segmenter` configuration to the runtime model resolution chain
  in `service/runtime.py`. Resolution order: explicit `--segmenter` /
  request field → `MSOCR_SEGMENTER_MODEL_PATH` env → default
  (stock blla for backward compatibility). The recognizer resolution chain
  is unchanged.
- The segmenter runs after Phase 1→2→3 isolation (on the deskewed fragment
  image, not the raw plate) and before `run_htr_service`. Output is a
  Kraken `Segmentation` object fed to the existing recognizer.
- Add a `train-seg-remote` CLI command mirroring `train-remote` for the
  chosen architecture, reusing `runpod_runner.py`. Do not add a local
  `train-seg` command until local GPU is available; segmentation training
  is heavier than recognition fine-tuning.
- Add a `segment-smoke-check` CLI command mirroring
  `runtime-smoke-check`, validating segmenter resolution and optionally
  running on a held-out image.
- Keep `extract-lines` (classical CV) as a `--segmenter row_bands` fallback
  in the runtime, so a failed learned segmenter degrades to the
  deterministic path instead of crashing.
- Update `docs/` with the new commands and the resolution chain.
- Do not touch the annotation UI. The segmenter's output is a *suggestion*
  to the annotator when used in the UI; it is never trusted ground truth
  (same rule as the existing `--propose-lines` flag:
  `training_eligible: False`).

**Exit criteria:** end-to-end HTR runs on a held-out fragment photo with
no human in the loop, using the learned segmenter + existing recognizer;
CER on the held-out set is recorded; `row_bands` fallback still works via
`--segmenter row_bands`.

---

## 6. Data requirements (common to all architectures)

| Requirement | Floor | Target | Source |
|---|---|---|---|
| Annotated plates (PAGE XML) | 40 | 80+ | Phase 1 annotation |
| Lines per plate (avg) | 5 | 10+ | Existing C2AV material |
| Total line samples | 200 | 600+ | Derived from plates |
| Held-out plates (never trained) | 6 | 10 | Phase 2 frozen set |
| Fragment-size diversity | small + medium + large represented in both train and held-out | same | Phase 1 plate selection |
| Damage diversity | holes, faint ink, split rows all represented in held-out | same | Phase 2 held-out selection |

These are the *minimum* to make any architecture's training meaningful. The
existing 12-plate / 154-line corpus is below every floor and must not be
used to train a segmenter.

---

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| orli fragment behavior is unknown; fine-tuning may fail | Phase 0 email; Phase 3 picks YOLO-OBB by default if no endorsement |
| 40 plates may still be too few for any architecture | Phase 2 harness reports zero-shot baselines first; if stock blla + `row_bands` both beat every trained candidate, stop and scale annotation further before retrying |
| YOLO bbox loses line curvature, hurting Sogdian recognition | Phase 2 end-to-end CER metric catches this; if bbox→recognizer CER is worse than `row_bands`, the bbox paradigm is rejected on evidence, not theory |
| Reading order on split-row fragments is unsolved by any segmenter | Accepted — `row_id` + RTL fragment order stay human-confirmed msocr metadata (locked decision §2.5) |
| RunPod cost for segmentation training is higher than recognition fine-tuning | Reuse `runpod_runner.py`; cap epochs; record cost per Phase 3/4 run |
| A learned segmenter that fails silently on a new fragment is worse than no segmenter | Phase 5 integration keeps `--segmenter row_bands` fallback; segmenter confidence threshold gates automatic acceptance |
| Annotation throughput is the bottleneck, not architecture choice | Phase 1 tracks plates/week; if throughput is too low, stop and reassess before committing to Phase 3 |

### Open questions for the maintainer email (Phase 0)

1. Has orli been evaluated on fragmentary / non-page inputs (small
   fragments, sparse lines, partial pages)? Base model card and paper
   describe full-page training; no fragment benchmarks found.
2. Does he have a view on orli's autoregressive baseline regression vs a
   bbox object detector (D-FINE / YOLO-OBB) as the starting point for
   fragment line detection, before investing in fine-tuning either?

---

## 8. What this plan deliberately does NOT do

- Does not pick an architecture. Phase 2 + Phase 0 decide.
- Does not write any code. No `train-seg-remote`, no `segment-smoke-check`,
  no segmenter resolution chain, no format converters — until Phase 5 (and
  Phase 3/4 training converters, which are small and gated).
- Does not touch the recognizer. `ketos train`, `KetosTrainer`,
  `walk_style_group`, `evaluate`, and the runtime recognition path are
  unchanged.
- Does not touch the annotation UI. The existing React SPA produces the
  PAGE XML this plan needs; no UI work is in scope.
- Does not replace `row_bands`. It stays as fallback and bootstrap.
- Does not model `row_id` or gaps in the segmenter. Those stay msocr
  metadata.
- Does not add Tesseract, OCRmyPDF, Surya, or any non-Kraken recognizer.
  The recognizer stays Kraken; only the segmenter is new.
- Does not commit to RunPod for segmentation training. Local GPU is an
  option if available; RunPod is the default via existing infra.

---

## 9. Success criteria

The plan succeeds when:

1. A learned segmenter, trained from the project's own PAGE XML annotation,
   runs end-to-end on a held-out fragment photo and produces per-line
   geometry that, fed to the existing Kraken recognizer, achieves lower
   end-to-end CER than the `row_bands` classical-CV baseline.
2. The `row_bands` fallback still works for plates where the learned
   segmenter fails or for annotation bootstrap.
3. The phase-2 evaluation harness can be re-run on any future candidate
   (orli v2, a new YOLO release, a D-FINE update) without rework.
4. Annotation throughput for the remaining C2AV plates has measurably
   accelerated (even a mediocre learned segmenter that suggests 70% of
   baselines correctly saves annotator time vs drawing every line by hand).

The plan fails (and should be revised) if:

- Phase 2 shows stock blla and `row_bands` both beat every trained
  candidate on end-to-end CER. Revision: scale annotation beyond 80 plates
  before retrying; do not lower the bar.
- Phase 0 reveals (from the maintainer) that orli is known to fail on
  fragments and no detection paradigm is recommended. Revision: reassess
  whether a learned segmenter is viable at this corpus scale, or invest in
  automating `expected_lines` estimation instead (a small regressor on
  fragment height / mean line spacing).
- Annotation throughput in Phase 1 is too low to hit 40 plates in a
  reasonable window. Revision: stop, diagnose the bottleneck (UI speed,
  annotator availability, image quality), and fix that before continuing.