# Fragmented Manuscript Model Training Plan

> Train a damage/lacuna detection model for fragmented Turfan/Sogdian manuscript
> folios, then feed its output as a gating mask into the Kraken OCR pipeline.

**Status:** research + plan, no implementation yet.
**Date:** 2026-07-09 (revised 2026-07-09 after v2 verification)
**Owner:** msocr

> **Companion doc:** `docs/damage-zone-model-plan-v2.md` independently re-verifies
> the literature and corrects one breaking Kraken 7.0 tooling assumption. This
> revision folds v2's corrections in and adds a `ketos orli_train` finding v2
> missed. Verification log:
> - Kraken 7.0.2 installed locally; `ketos segtrain --valid-regions` confirmed
>   **removed** — replaced by `--resize [add|union|both|new|fail]` + top-level
>   `--config FILENAME` YAML (verified via `ketos segtrain --help` and
>   `ketos --help`, 2026-07-09).
> - `ketos orli_train` exists in 7.0.2 — Kraken-native object-detection trainer
>   from XML facsimile files. v2 did not flag this; added below as Approach 2.5.
> - Borkar & Smith (arXiv:2407.00250) four claims all CONFIRMED: model is TrOCR
>   (not "TrCro"); synthetic lacunae raise lacuna restoration 5.6%→65.85%;
>   logistic regression on min(log-probability) flags lacuna lines 53.25% and
>   other error lines 84.12%; attention weights not significantly better.

---

## 1. Motivation

Turfan/Sogdian folios are physically damaged: lacunae, holes, stains, tears,
foxing, charred patches. Today msocr handles this only at **inference time**
via a manual `kraken segment -m mask.png` where a human draws the mask, and
via `DamageZone` polygons annotated in the frontend (`AnnotateEditor.tsx`).
There is **no model** that detects damage on new, unseen folios. Kraken HTR
training (`ketos train`) uses only baselines + transcripts, so damage is
invisible to the recognizer — it just transcribes noise across gaps.

Goal: a separate model that produces a damage mask per folio, fed into the
Kraken HTR pipeline so that:

1. Kraken `segment` skips line detection in damaged bands (`-m mask.png`).
2. Kraken recognition never attempts to read across lacunae, lowering CER
   on fragmentary folios and giving cleaner training data for the next HTR
   fine-tune cycle.

This is a two-stage pipeline: **damage-detection model → mask → Kraken HTR**.

---

## 2. Current state (codebase recon)

Verdict: **no damage model exists in msocr.** Confirmed by recon (exp-1).

| Component | Current state | Gap |
|---|---|---|
| Training entrypoint | `msocr/training/` has only Kraken `ketos` wrappers (`ketos_trainer.py`, `ketos_trainer_api.py`, `runpod_runner.py`, `orchestrator.py`, `dump_preds.py`). No `train-damage`/`train-region`/`train-layout`. | New subcommand + module |
| Pre-HTR inference hook | `msocr/models/inference.py` does `manuscript_area.detect()` (union bbox of ink CCs) only. `runtime.py:run_htr_service()` calls `predict_line()` with `segmentation_type="baseline"`. No damage mask. | Insert damage prediction step |
| Ground-truth labels | Manifests (`data/manifest.py`, `data/manifests/*.json`) carry baselines + transcripts only. DamageZone polygons live in annotation session state (`session_manager.py`), not as training labels. | Export DamageZone polygons as segmentation masks for training |
| CLI | 15 subcommands (`htr`, `train`, `preprocess`, `extract-lines`, `isolate-fragments`, `binarize-fragments`, `deskew-fragments`, `api`, `annotation-api`, `demo`, `runtime-smoke-check`, `train-remote`, `evaluate`, `dump-preds`, `annotate`). None for damage/layout. | New `train-damage` / `detect-damage` subcommands |
| Docs | `kraken-fragmentary-manuscripts.md:30` explicitly: *"No per-pixel 'don't-care' zone exists in Kraken's seg or rec training."* `fix-a-v9-fragment-pipeline.md:143-146`: manual `segment(mask=…)` at inference time. | Acknowledged, not implemented |

Grep in `*.py` for `DamageZone`, `damage_zone`, `damage_region`, `train_damage`,
`train_region`, `train_layout`, `yolo`, `unet`, `mask.*rcnn`, `lacuna`, `hole`,
`stain`: **zero matches**.

So DamageZone is currently a **human annotation convention**, not a model
target.

---

## 3. Literature review

### 3.1 Does Kraken itself train on regions/DamageZone?

Kraken 7.0.2 has **three** training modes, not two (verified locally):

- **`ketos train`** — recognition/HTR. Uses only baselines + transcripts.
  Region polygons are NOT training targets.
- **`ketos segtrain`** — baseline-labeling layout segmentation. Pixel model
  trained on PAGE XML baselines + region polygons. **Kraken 7.0 removed
  `--valid-regions`/`--merge-regions`.** Class handling is now via
  `--resize [add|union|both|new|fail]` (controls how training-data classes
  combine with a loaded model's classes) and a top-level `--config FILENAME`
  YAML file for class mapping. DamageZone *can* still be a region class.
- **`ketos orli_train`** — **object detection** from XML facsimile files
  (newly confirmed in 7.0.2; v2 doc did not flag this). This is the
  Kraken-native equivalent of the YALTAi idea — region detection as
  bounding boxes rather than pixel classification. Relevant for DamageZone
  because object detection handles rare classes better than pixel seg.

Refs: `ketos --help` / `ketos segtrain --help` / `ketos orli_train --help`
(local, kraken 7.0.2, 2026-07-09),
[Kraken segtrain docs](https://kraken.re/main/training/segtrain.html),
[Kraken ketos docs](https://kraken.re/6.0.0/training/ketos.html),
[Digital Orientalist tutorial](https://digitalorientalist.com/2023/11/03/11400/),
[eScriptorium train docs](https://escriptorium.readthedocs.io/en/latest/train/).

**Implication:** a minimal approach is to fine-tune Kraken's own `blla.mlmodel`
via `ketos segtrain` (YAML class mapping, NOT `--valid-regions`) and reuse the
Kraken ecosystem (no new model framework needed). The trade-off is class
imbalance (DamageZone is rare). `ketos orli_train` is an alternative
Kraken-native route that may handle rare DamageZone better as bounding boxes.

### 3.2 Separate-model approaches

| Approach | Ref | Notes |
|---|---|---|
| **PLM-SegFormer** (SegFormer fine-tune, 5 damage classes on Sanskrit palm leaves) | Wang et al. 2024, *Heritage Science* — [doi](https://doi.org/10.1186/s40494-023-01125-w), [GitHub](https://github.com/Ryan21wy/PLM_SegFormer) | 70.1% mHit, 51.2% mIoU, 10,064 folios in 12h. **Closest published analogue to what we want.** |
| **MADAM** (Mask R-CNN / U-Net / DeepLabv3+ on flood-damaged manuscripts) | Carcagnì et al. 2024, AIUCD 2025 — [pdf](https://aiucd2025.dlls.univr.it/assets/pdf/papers/118.pdf) | Character-level instance seg; explicitly excludes damage from OCR. |
| **MTEM** (multispectral thresholding + energy min for parchment/ink/holes on Dead Sea Scrolls) | IJDAR 2025 — [link](https://link.springer.com/article/10.1007/s10032-025-00564-4) | Hole detection specifically; needs multispectral input (we don't have). |
| **YALTAi** (YOLOv5 for region detection incl. DamageZone) | Clérice 2022, *JDMDH* — [doi](https://doi.org/10.46298/jdmdh.9806), [HF dataset](https://huggingface.co/datasets/biglam/yalta_ai_segmonto_manuscript_dataset) | Object detection; **only 21 DamageZone instances** in the whole dataset. Near-zero AP on DamageZone alone. |
| **AutoHDR** (OCR-assisted damage localization → restoration → re-OCR; 46.83% → 84.05%) | Zhang et al. 2025 — [arXiv](https://arxiv.org/html/2507.05108) | Full two-stage pipeline precedent. |
| **DocRevive** (YOLOv9c occlusion detector → LM → diffusion editing) | 2025 — [arXiv](https://arxiv.org/html/2604.10077v2) | Damage as object detection, pre-OCR gating. |
| **Lacuna detection via log-probability** (TrOCR flagging lacuna lines w/o damage training) | Borkar & Smith 2024, ICDAR 2024 Workshop on Computational Paleography — [arXiv:2407.00250](https://arxiv.org/abs/2407.00250) | **Numbers verified** (lib-2, 2026-07-09): model is TrOCR (not "TrCro"); synthetic-lacuna training raises lacuna restoration **5.6%→65.85%**; logistic regression on min(log-probability) flags lacuna lines **53.25%** and other-error lines **84.12%** *without looking at the image*; attention weights not significantly better. Feature already exposed by Kraken's `BaselineOCRRecord.confidences`. |
| **Color-space segmentation** (GMM in CIELab/CIELuv for foreground/background/degradation/annotation) | Hanif et al. 2023, *PLOS ONE* | Non-deep; effective for stains/bleed-through/mold. |
| LayoutLMv3 / DiT (Microsoft) | [arXiv](https://arxiv.org/abs/2204.08387), [arXiv](https://arxiv.org/abs/2203.02378) | Modern document layout; not manuscript damage. |

### 3.3 Turfan / Sogdian / Central Asian specifically

**No published computer-vision work on damage detection for Sogdian/Turfan
manuscripts was found.** The Berlin Turfan collection (~40,000 fragments) is
digitized and catalogued (Reck 2006–2018), but computational work focuses on
fragment reassembly (LLMCO4MR, ECCV 2024) and handwriting-style matching
for Dunhuang (npj Heritage Science 2025). Reck describes Turfan fragments as
"very badly damaged" with identification being "very difficult."

**This is a genuine research gap.** A working damage detector for Sogdian
fragments would be a legitimate contribution.

**Adjacent finding (v2):** the BBAW/MPIWG "Turfan Studies" project has
imaged some Berlin Turfan material with phase-contrast X-ray and
hyperspectral imaging. Not accessible here, but if a multispectral scan of
our specific fragments ever surfaces, it would sidestep the RGB-only
limitation.

### 3.4 Data requirements

Realistically **a handful of folios cannot train a damage model from scratch**.
YALTAi and CATMuS confirm DamageZone is the rarest region class by orders of
magnitude (21 and 13 instances vs thousands of MainZone). **v2 sharpens
this**: PLM-SegFormer's full Sanskrit palm-leaf corpus is only **2.9%
damage pixels** despite being "seriously affected" overall — expect Sogdian
DamageZone to be in the same single-digit-percent pixel range, not just rare
in instance count. From-scratch training would overfit catastrophically.

Practical route is **transfer learning**:

- Kraken `blla.mlmodel` fine-tune: ~30–50 pages for book-specific tasks
  (Digital Orientalist), ~130 docs from scratch for ~41% accuracy
  (Digital Intellectuals).
- SegFormer/YOLO fine-tune from COCO/ImageNet pretraining: works with
  fewer annotated examples; PLM-SegFormer is the template.
- <50 folios → prefer Kraken fine-tuning. ≥50 → dedicated seg model becomes
  viable and likely better.

### 3.5 Two-stage "detect damage → mask → HTR" precedents

Documented pipelines: AutoHDR, DocRevive, EpiText-Hanja-OCR
(`[MASK1]`/`[MASK2]` tokens), modular early-printed-books pipeline (Kraken
seg → OCR → ByT5 post-correction, MDPI Electronics 2025). The architecture
is not novel — it's the right instinct.

### 3.6 Additional finds (from v2 verification)

- **Indiscapes** (arXiv:1912.07025, instance seg for Indic manuscripts):
  explicitly reports "Physical degradations" as their hardest-to-parse region
  class — independent third-system confirmation of the same DamageZone
  failure mode YALTAi and PLM show.
- **MapSAM / MapSAM2** (arXiv:2411.06971 / 2510.27547): SAM/SAM2 adapted to
  historical maps via LoRA-style adapter layers on a frozen encoder. Closer
  methodological analogue to our situation than PLM-SegFormer — foundation
  model, degraded historical domain, few labels. Documented fallback if
  SegFormer from-scratch fine-tuning underperforms at our data scale.

---

## 4. Approaches for msocr

| # | Approach | Needs pixel-level DamageZone labels? | Effort | GPU? | Output |
|---|---|---|---|---|---|
| 1 | **Kraken `--mask` at inference** (render DamageZone polygons → binary mask → `kraken segment -m mask.png`) | No (uses existing polygons) | Minimal | No | Mask gate on already-annotated folios |
| 2 | **`ketos segtrain` fine-tune** (YAML class mapping, NOT `--valid-regions`) | Yes | Medium | Yes | Auto-mask on new folios, pixel seg |
| 2.5 | **`ketos orli_train`** (Kraken-native object detection, new in 7.0.2) | Yes (boxes from polygons) | Medium | Yes | Auto-detect DamageZone as boxes; rare-class friendlier than pixel seg |
| 3 | **Dedicated SegFormer / YOLOv8-seg / SAM2-adapter** (PLM-SegFormer or MapSAM2 template), fed as mask before Kraken HTR | Yes | Higher | Yes (transfer) | Best long-term mask; publishable |
| **4 (new, parallel track)** | **Lacuna-aware `ketos train`** (synthetic-lacuna augmentation of existing line GT) + per-line log-prob flagging | **No** | **Low** | Optional | Per-line confidence signal; complements 1–3; needs zero new annotation |

**Approach 4 (from v2, Borkar & Smith–verified):** does not produce a spatial
mask — it produces a per-line signal telling you a line probably crosses
damage, using only the transcription GT you already have. It cannot tell you
*where* in a line the gap is, only *that* the line is suspect. Free to build
in parallel; gives a sanity check against the vision-based mask (any folio
the vision mask marks undamaged but Approach 4 flags low-confidence is worth
a second look, and vice versa).

Refs:
- Kraken mask: [page segmentation docs](https://kraken.re/6.0.0/advanced/segmentation.html)
- `ketos segtrain` (7.0): [segtrain docs](https://kraken.re/main/training/segtrain.html) + local `ketos segtrain --help`
- `ketos orli_train`: local `ketos orli_train --help` (kraken 7.0.2)
- PLM-SegFormer: [Heritage Science 2024](https://doi.org/10.1186/s40494-023-01125-w)
- YALTAi: [JDMDH 2023](https://doi.org/10.46298/jdmdh.9806)
- MapSAM2: [arXiv:2510.27547](https://arxiv.org/pdf/2510.27547)
- Borkar & Smith (Approach 4 basis): [arXiv:2407.00250](https://arxiv.org/abs/2407.00250) — numbers verified
- HDR28K (degradation simulation for augmentation): [AAII 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33016/35171)

---

## 5. Recommended roadmap

### Phase 0 — Mask gate (unchanged, still correct)

Wire Approach 1 into `msocr/service/runtime.py` and `msocr/cli.py htr`:

- For folios with annotated DamageZone, render polygons as a binary mask
  (white = damaged, black = valid) at the page image's dimensions.
- Pass `mask=…` to `kraken segment`.
- No training, no new model. Immediate payoff on already-annotated folios.
- `segment -m` is untouched in Kraken 7.0 — no version issue here.

Acceptance: `msocr htr` on a folio with a DamageZone polygon no longer
emits baselines/transcripts inside the damage area.

### Phase 0.5 — Lacuna-aware recognizer (new, parallel to Phase 0)

Per v2 §1.2 + verified Borkar & Smith numbers.

- Write a synthetic-lacuna augmentation pass over your existing line-image
  training set: randomly blank contiguous spans (mimicking hole/tear
  geometry — not random pixels) in a fraction of training lines, with the
  ground-truth transcript marking the span (Leiden-convention bracket
  token, matching Borkar & Smith's setup, adaptable to the Sogdian charset).
- Fine-tune the existing Sophro Mhiro-based Kraken recognizer on the
  augmented set via ordinary `ketos train` — **no CLI version issue here**;
  this is recognition training, not segmentation, so the 7.0
  `--valid-regions` removal doesn't apply.
- At inference, log per-line mean/min confidence from
  `BaselineOCRRecord.confidences` (already computed) and flag lines below a
  threshold calibrated against the held-out damage eval set (§5).
- Acceptance: flagged-line precision/recall against the same held-out
  damaged-folio set built for Phase 2, benchmarked against the
  53.25%/84.12% Borkar & Smith numbers as a sanity floor — we should beat
  their generic historical-document baseline given Sogdian's narrower domain.
- **Caveat (v2 §6):** synthetic lacunae won't match real damage geometry.
  Treat Phase 0.5's signal as a coarse triage tool, not a substitute for the
  vision-based mask.

### Phase 1 — Data consolidation (enables Phase 2 & 3, bottleneck)

- Export DamageZone polygons from all annotated PAGE XML sessions
  (`session_manager.py` already holds them in v2 state) into a
  segmentation-dataset format.
- Decide damage subclasses: `DamageZone:hole`, `:stained`, `:charred`,
  `:foxed`, `:discoloured` (per `plans/2026-06-22-segmentation-pipeline-plan.md:315`)
  or collapse to a single `DamageZone` to fight class imbalance.
- Establish a held-out damage evaluation set (folios with damage, transcribed
  by a human, with CER computed with and without the mask).
- **v2 §5 addition:** do NOT randomly hold out folios. Deliberately
  over-sample DamageZone into the eval split (YALTAi authors had to do this
  themselves). Stratify by damage severity (light foxing vs. holes vs.
  charring) if tracking subclasses at all, even if the training run collapses
  to a single `DamageZone` class per the v1 recommendation.

Acceptance: a `data/damage/` directory with one mask image per folio and a
manifest listing damage subclasses + a stratified eval split.

### Phase 2 — Kraken seg fine-tune (CORRECTED for Kraken 7.0.2)

**v1's command is dead on Kraken 7.0.2** — `--valid-regions`/`--merge-regions`
removed. Build a YAML experiment file and pass it via top-level `--config`:

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
  load: blla.mlmodel          # base model to fine-tune from
  checkpoint_path: seg_checkpoints
  weights_format: safetensors
  # class mapping now in YAML, NOT a CLI flag. Verify exact key name
  # (region_class_mapping vs line_class_mapping vs combined) against
  # `ketos segtrain --help` / kraken.re/main/training/segtrain.html
  # before the first run — five-minute check that avoids a failed run.
  region_class_mapping:
    - ['DamageZone', 3]
    - ['MainZone', 4]
    # add remaining Segmonto/DamageZone-subclass types as needed,
    # or merge everything else into one background class
```

```
ketos --config damage_segtrain.yaml segtrain
```

Also `--training-files`/`--evaluation-files` were renamed to
`--training-data`/`--evaluation-data` in 7.0; checkpoints are now Lightning
`.ckpt` with `--resume` (checkpoint-authoritative) and `--load` (fresh run
from weights). Verify Phase 2 acceptance criterion (v1 said IoU ≥ 0.4)
against what `ketos segtest` actually reports on 7.0 — it may now emphasize
baseline-detection F-scores over region IoU by default.

Acceptance: predicted DamageZone IoU (or `segtest`'s default region metric)
on held-out folios meets a threshold calibrated against PLM-SegFormer's
0.51 mIoU on palm leaves (single-class DamageZone should be in reach).

### Phase 2.5 — `ketos orli_train` (Kraken-native object detection, alternative to Phase 2)

`ketos orli_train` exists in 7.0.2 — "Trains an object detection model from
XML facsimile files." This is the Kraken-native equivalent of the YALTAi
idea: detect DamageZone as **bounding boxes** rather than pixel seg. Object
detection handles rare classes better than pixel classification (which is
exactly the DamageZone failure mode YALTAi, CATMuS, and Indiscapes all
report). Convert DamageZone polygons → axis-aligned or rotated boxes, train
via `orli_train`, predict boxes on new folios, rasterize boxes → mask →
feed into Phase 0 gate.

**Decision point:** run Phase 2 and Phase 2.5 in parallel on the same
held-out eval set; pick whichever wins on DamageZone recall at acceptable
precision. If Phase 2 plateaus on DamageZone specifically (expected, given
the imbalance), Phase 2.5 is the likely winner. v2 did not flag
`orli_train`; this is a new option.

### Phase 3 — Dedicated damage model (Approach 3, with SAM2 fallback)

- Only if Phase 2 / 2.5 plateau and ≥50 folios with damage are annotated.
- Fine-tune SegFormer (PLM-SegFormer template) or YOLOv8-seg on the Phase 1
  dataset; export model, add `msocr detect-damage` subcommand, hook output
  into the Phase 0 mask gate.
- Consider augmentation via degradation simulation (HDR28K approach) if
  DamageZone instances stay rare.
- **v2 fallback (MapSAM2):** if SegFormer from-scratch fine-tuning
  underperforms at our data scale, switch to SAM2 + LoRA-style adapter
  layers on a frozen SAM2 encoder. MapSAM2 is designed for exactly our
  constraint shape (foundation model, historical/degraded domain, few
  labels) and has a lower annotation floor than full fine-tuning.
- Publishable result — fills the Sogdian/Turfan damage-detection gap.

Acceptance: dedicated model strictly beats Phase 2/2.5 on the held-out
damage eval (IoU + CER-downstream), justifying the extra stack.

---

## 6. Pipeline integration point

```
page image
   │
   ├─► [Phase 0/2/3 damage model] ──► damage mask (white=damage)
   │                                        │
   ▼                                        ▼
kraken segment -bl -m mask.png  ──►  baselines (skipping damage)
   │
   ▼
kraken rec  ──►  transcripts
   │
   ▼
output formats (JSON/Markdown)
```

Insertion point in code:
- `msocr/models/inference.py` after `manuscript_area.detect()` (~line 102)
  — call damage model / render DamageZone polygons → produce mask.
- `msocr/service/runtime.py:run_htr_service()` — thread the mask into
  `OCRModel.predict_line(..., mask=mask)`.
- `msocr/cli.py htr` — accept `--damage-mask <path>` or auto-derive from
  sibling PAGE XML if present.

---

## 7. Risks / unknowns

- **Class imbalance.** DamageZone is the rarest class everywhere it's been
  measured (YALTAi 21 instances, CATMuS 13, Indiscapes "Physical
  degradations" hardest class). **v2 sharpens:** PLM-SegFormer's full corpus
  is only 2.9% damage pixels — expect Sogdian DamageZone in the same
  single-digit-percent pixel range, not just rare in instance count.
  Single-class collapse + class weighting or augmentation is likely
  mandatory. Phase 2.5 (object detection) may sidestep this where pixel seg
  fails.
- **Label noise.** DamageZone drawn by a human is subjective (where does a
  stain end?). Need ≥2 annotators or a single trusted annotator with a
  written guideline (`ANNOTATION.md` already seeds this).
- **Multispectral gap.** MTEM (DSS) and MADAM rely on multispectral input
  we don't have. Our model has to work on RGB scans. (Caveat: BBAW/MPIWG
  Turfan Studies has hyperspectral/X-ray imaging of some Berlin Turfan
  material — not accessible here, but worth knowing if our specific folios
  ever get scanned that way.)
- **Two-model inference cost.** Adding a damage model before Kraken HTR
  roughly doubles inference latency per folio. Acceptable for offline
  training-data prep, possibly not for the live API path.
- **No Sogdian precedent.** Phase 3 is research, not engineering — expect
  iteration.
- **Kraken 7.0 API churn is ongoing (v2 §6).** The class-mapping migration
  happened recently enough that most online Kraken guidance still shows
  the old `--valid-regions` syntax. Budget time for docs/reality mismatches
  beyond just this one flag when running Phase 2.
- **Approach 4's synthetic lacunae won't match real damage geometry (v2 §6).**
  Treat Phase 0.5's confidence signal as a coarse triage tool, not a
  substitute for the vision-based mask — use it to prioritize which folios
  get human review first, not as ground truth for Phase 1's dataset.

---

## 8. Open questions (need user decision before implementation)

1. **Damage subclasses or single class?** Subclasses
   (`hole`/`stained`/`charred`/`foxed`/`discoloured`) give richer signal but
   explode imbalance. Default recommendation: single `DamageZone` for v1,
   subclasses for v2.
2. **Where does the damage model run?** Local CPU for offline training-data
   prep; RunPod GPU (existing `train-remote` infra) for training; live API
   path? Default recommendation: offline-only for v1.
3. **Eval dataset.** How many held-out damaged folios can we reserve, and who
   transcribes them? Drives Phase 2 vs Phase 3 decision. **v2 §5:**
   deliberately over-sample damage into the eval split, do not random-split.
4. **(v2 §7.4) Build Phase 0.5 now, in parallel with Phase 1?** Phase 0.5
   reuses the existing HTR pipeline (`ketos_trainer.py`) with no new
   annotation format, so it could ship before Phase 0's mask gate (which
   needs the polygon-to-raster rendering step built first). Recommend: yes,
   start Phase 0.5 immediately — cheapest independent signal.
5. **(v2 §7.5) Confirm YAML key names** for `ketos segtrain` region class
   mapping on installed Kraken 7.0.2 before Phase 2 implementation — direct
   `ketos segtrain --help` / doc check, five minutes, avoids a wasted
   training run.
6. **(new) Phase 2 vs Phase 2.5 — pixel seg or object detection?** Both are
   Kraken-native in 7.0.2. Object detection (`orli_train`) may handle rare
   DamageZone better. Recommend: run both on the same eval split, pick the
   winner. Decision can wait until Phase 1 data exists.

---

## 9. Status

- [x] Background research (sections 2–3)
- [x] Approaches identified (section 4; folded in v2's Approach 4 + new 2.5)
- [x] Roadmap drafted (section 5; Phase 0.5 added, Phase 2 corrected for 7.0,
      Phase 2.5 added, Phase 3 SAM2 fallback added)
- [x] v2 literature re-verified; Borkar & Smith numbers confirmed (lib-2)
- [x] Kraken 7.0.2 breaking change verified locally (`--valid-regions` gone)
- [ ] Phase 0 implementation (mask gate in `runtime.py` + `cli.py`)
- [ ] Phase 0.5 implementation (synthetic-lacuna `ketos train` + confidence flag)
- [ ] Phase 1 data consolidation (export DamageZone polygons → masks; stratified eval)
- [ ] Phase 2 Kraken seg fine-tune (YAML config, NOT `--valid-regions`)
- [ ] Phase 2.5 `ketos orli_train` object-detection alternative
- [ ] Phase 3 dedicated damage model (if Phase 2/2.5 plateau)

No code written yet. Next action: user confirms open questions in §8, then
Phase 0 + Phase 0.5 are the first implementation tasks (parallel, independent).

---

## 10. References

### Damage / lacuna detection on manuscripts
- PLM-SegFormer (Wang et al. 2024, *Heritage Science*) — [doi](https://doi.org/10.1186/s40494-023-01125-w), [GitHub](https://github.com/Ryan21wy/PLM_Segformer)
- MADAM (Carcagnì et al. 2024, AIUCD 2025) — [pdf](https://aiucd2025.dlls.univr.it/assets/pdf/papers/118.pdf)
- MTEM for Dead Sea Scrolls (IJ DAR 2025) — [link](https://link.springer.com/article/10.1007/s10032-025-00564-4)
- YALTAi (Clérice 2022, *JDMDH*) — [doi](https://doi.org/10.46298/jdmdh.9806), [HF dataset](https://huggingface.co/datasets/biglam/yalta_ai_segmonto_manuscript_dataset)
- Indiscapes — [arXiv:1912.07025](https://arxiv.org/pdf/1912.07025)
- Borkar & Smith, "Mind the Gap" (ICDAR 2024 Workshop) — [arXiv:2407.00250](https://arxiv.org/abs/2407.00250) — numbers verified

### Two-stage "detect → mask → OCR" pipelines
- AutoHDR — [arXiv](https://arxiv.org/html/2507.05108)
- DocRevive — [arXiv](https://arxiv.org/html/2604.10077v2)
- Modular early-printed-books pipeline (MDPI Electronics 2025) — [link](https://www.mdpi.com/2079-9292/14/15/3083)
- EpiText-Hanja-OCR — [HuggingFace](https://huggingface.co/donghyun95/EpiText-Hanja-OCR)

### Foundation-model adaptation for historical / degraded domains
- MapSAM — [arXiv:2411.06971](https://arxiv.org/pdf/2411.06971)
- MapSAM2 — [arXiv:2510.27547](https://arxiv.org/pdf/2510.27547)
- LayoutLMv3 — [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
- DiT — [arXiv:2203.02378](https://arxiv.org/abs/2203.02378)

### Augmentation / restoration
- HDR28K (DiffHDR, AAAI 2025) — [link](https://ojs.aaai.org/index.php/AAAI/article/download/33016/35171)
- Color-space GMM segmentation (Hanif et al. 2023, *PLOS ONE*) — [link](https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0282142&type=printable)

### Kraken / eScriptorium tooling (Kraken 7.0.2 verified)
- `ketos --help`, `ketos segtrain --help`, `ketos orli_train --help` (local, 2026-07-09)
- Kraken segtrain docs (current) — [kraken.re/main/training/segtrain.html](https://kraken.re/main/training/segtrain.html)
- Kraken ketos docs — [kraken.re/6.0.0/training/ketos.html](https://kraken.re/6.0.0/training/ketos.html)
- Kraken page segmentation docs (`-m mask`) — [kraken.re/6.0.0/advanced/segmentation.html](https://kraken.re/6.0.0/advanced/segmentation.html)
- Kraken releases — [github.com/mittagessen/kraken/releases](https://github.com/mittagessen/kraken/releases)
- eScriptorium train docs — [escriptorium.readthedocs.io/en/latest/train/](https://escriptorium.readthedocs.io/en/latest/train/)
- Digital Orientalist tutorial — [digitalorientalist.com/2023/11/03/11400/](https://digitalorientalist.com/2023/11/03/11400/)
- Digital Intellectuals blog — [digitalintellectuals.hypotheses.org/3844](https://digitalintellectuals.hypotheses.org/3844)

### Turfan / Sogdian context
- Reck, Berlin Turfan catalogue — [Cambridge chapter](https://www.cambridge.org/core/books/studies-on-the-iranian-world-before-islam/work-in-progress-the-catalogue-of-the-buddhist-sogdian-fragments-of-the-berlin-turfan-collection/35D42AB44F429378E5DDDF85F52684F0)
- BBAW/MPIWG Turfan Studies (hyperspectral/X-ray) — [mpiwg-berlin.mpg.de](https://www.mpiwg-berlin.mpg.de/feature-story/objects-revealed-unveiling-astral-manuscripts-berlin-turfan-collection)
- LLMCO4MR fragment reassembly (ECCV 2024) — [ecva.net](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09585.pdf)
- Dunhuang handwriting-style matching (npj Heritage Science 2025) — [nature.com](https://preview-www.nature.com/articles/s40494-025-02078-y)