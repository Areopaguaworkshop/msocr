# Sogdian OCR Segmentation Pipeline — Implementation Plan

Date: 2026-06-22
Status: Draft (pending user confirmation)
Supersedes: current `manuscript_area.py` union-bbox + default-Kraken BLLA approach
Aligns with: `2026-06-17-msocr-training-pipeline-design.md` (HTR fine-tuning track — referenced, not duplicated)

---

## 1. Problem

Current pipeline:
- `manuscript_area.py` → union bbox of all ink CCs ≥ min_area, padded 20px. No region typing, no density model, no marginalia separation. Marginalia pulls ROI outward.
- `kraken_blla.py` → default `blla.mlmodel` (one region type `'text'`, one line type `'default'`). Cannot distinguish MainTextZone from MarginalTextZone from NumberingZone.
- `preprocessor.py` → OpenCV denoise/contrast/skew/binarize. Dead `skimage` imports. Whole-page deskew applied to multi-fragment scans (wrong: each fragment has its own skew).
- `row_bands.py` → row extraction by CC clustering. Reusable for fragment isolation.

Goal: a 5-stage pipeline (fragment isolation → binarization → deskew → line+region segmentation → recognition) where segmentation is driven by a fine-tuned Kraken BLLA model trained on the user's annotated Sogdian pages, producing region-typed output (MainTextZone / MarginalTextZone / NumberingZone).

---

## 2. Verified Foundations (from lib-1 through lib-4)

| Claim | Source |
|-------|--------|
| `ketos segtrain -f xml *.xml` trains baselines + regions **jointly** in one model | kraken.re/5.1/ketos.html |
| Kraken 7 YAML config: `line_class_mapping`, `region_class_mapping` with SegmOnto labels | 7.0 release notes |
| Output: `.safetensors` (default) or `.mlmodel` (`--weights-format coreml`) | 7.0 release notes |
| `SegmentationTaskModel.load_model('x.safetensors')` — direct drop-in | api.md |
| eScriptorium PAGE XML ingested directly, no conversion | ketos.rst |
| `from kraken.binarization import nlbin` exists but is **deprecated** for BLLA | binarization.py, advanced.html |
| BLLA works on color/grayscale — binarization only needed for legacy bbox segmenter | advanced.html |
| `--suppress-baselines` / `--suppress-regions` for separate training | kraken.re/5.1/ketos.html |
| Fine-tune recognition: `ketos train --load existing.safetensors --resize new` | training_recognition.md |
| `ketos convert -o out.safetensors in.mlmodel` (no retrain) | 7.0 release notes |
| Region types in `seg.regions` are automatic from model metadata | api.html |
| Python API gives clean JSON: per-line text + baseline + region assignment | api.md |

Killed: SAM2 (promptable, not auto-detecting), YALTAi (archived, GPL), Swin-UNet+Mask R-CNN (no public code), PERO-OCR (no public training code — inference only).

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Turfan photo (dark bg, ruler, Munsell card, fragments)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
       Phase 1: CC fragment isolation
       (Sauvola coarse w=51 → CC → cluster by proximity → fragment rois)
                            │
       Phase 2: Binarization (geometry only, not pre-recognition)
       (fine Sauvola w=25 for CC/clustering; nlbin only if bleed-through)
                            │
       Phase 3: Per-fragment Hough deskew
       (extends existing correct_skew, applied per-fragment not per-page)
                            │
       Phase 4: Kraken BLLA segmentation (fine-tuned model)
       (SegmentationTaskModel.predict → seg.lines[] + seg.regions{})
                            │
       Phase 5: Recognition (fine-tuned HTR model)
       (RecognitionTaskModel.predict per line → text + confidence)
                            │
       Output: per-fragment PAGE XML + JSON with FRAGMENT_ID metadata
       (cross-fragment reading order = manual scholarly task, out of scope)
```

**Binarization note (lib-4 finding):** BLLA works on color/grayscale. We do **not** binarize before Phase 4. Binarization (Sauvola + optional nlbin) is used only in Phase 1/2 for CC-based geometry (fragment isolation, line-band clustering). Phase 4 receives the color/grayscale deskewed fragment.

---

## 4. Dependency-Ordered Phase Graph

```
Phase 1 ── validate ──> Phase 2 ── validate ──> Phase 3 ── validate ──> Phase 4 (training) ── gate ──> Phase 5 ──> Phase 6 (verification)
                                                                          │
                                                                          └── Phase 4b: recognition fine-tune (parallel, covered by existing 2026-06-17 plan)
```

Go/no-go gates between phases. Each phase produces a verifiable artifact before the next begins.

---

## 5. Phases

### Phase 1 — CC Fragment Isolation
**Goal:** Replace `manuscript_area.py` union-bbox with per-fragment ROI detection.

**Method:**
- Coarse Sauvola threshold (window=51) on grayscale → binary mask.
- `cv2.connectedComponentsWithStats` on mask.
- Filter components by area ≥ min_area (reuse `_ink_components` logic from `row_bands.py`).
- Cluster nearby components into fragments by spatial proximity (DBSCAN on component centroids, eps = typical inter-fragment gap).
- Sub-cm fragments flagged `FRAGMENT_TOO_SMALL` (not dropped — scholarly data).
- Output: list of fragment rois `(x, y, w, h, fragment_id)`.

**Files:**
- New: `msocr/segmentation/fragment_isolation.py`
- Reuse: `_ink_components`, `_clamp_roi` from `row_bands.py`
- Replace: `manuscript_area.py` (kept as fallback during transition)

**Validate (gate):**
- Run on `tmp/pdfs/ms_c2av_image19/page-19.png`.
- Visual check: each manuscript fragment gets its own roi, dark background + ruler + Munsell card excluded.
- Expected: ~18 manuscript rows per `line-segmentation-strategy.md` — confirm fragment count matches expected row count (or is close, allowing for merged/split fragments).
- Output: `tmp/phase1_fragments.png` (visualization) + `tmp/phase1_fragments.json` (roi list).

**Ponytail note:** No new abstraction. CC + DBSCAN is stdlib OpenCV + sklearn. No SAM2.

---

### Phase 2 — Binarization (Geometry Only)
**Goal:** Produce binary masks for downstream geometry (deskew angle detection, line-band clustering). Not fed to BLLA.

**Method:**
- Fine Sauvola threshold (window=25) per fragment → binary mask for geometry.
- `kraken.binarization.nlbin()` only for fragments with visible bleed-through (detected by histogram bimodality check). Deprecated but functional; CLI subprocess alternative if Python import unstable.
- Phase 4 (BLLA) receives the **color/grayscale** deskewed fragment, not the binary mask.

**Files:**
- New: `msocr/preprocessing/binarize.py` (Sauvola wrapper + nlbin fallback)
- Modify: `msocr/preprocessing/preprocessor.py` (remove dead `skimage` imports, redirect to new `binarize.py`)

**Validate (gate):**
- Run on Phase 1 fragment rois.
- Visual check: text strokes are connected (for deskew Hough), background is uniform.
- Bimodality check: each fragment's histogram has two clear peaks.

**Ponytail note:** Sauvola is `skimage.filters.threshold_sauvola` (stdlib). nlbin is one function import. No custom binarization class.

---

### Phase 3 — Per-Fragment Deskew
**Goal:** Correct each fragment's skew independently (multi-fragment photos have per-fragment skew).

**Method:**
- Hough transform on Phase 2 binary mask → detect dominant text line angle.
- Rotate fragment by negative of detected angle.
- Extend existing `correct_skew` in `preprocessor.py` to operate per-fragment instead of per-page.

**Files:**
- Modify: `msocr/preprocessing/preprocessor.py` (`correct_skew` → `correct_fragment_skew`)
- Reuse: existing Hough logic

**Validate (gate):**
- Run on Phase 2 binary masks.
- Visual check: text lines are horizontal (or RTL-appropriate) after rotation.
- Skew angle per fragment logged to `tmp/phase3_deskew.json`.

**Ponytail note:** Extends existing function. No new deskew algorithm.

---

### Phase 4 — Kraken BLLA Segmentation Fine-Tuning
**Goal:** Train a region-typed + baseline-typed Kraken segmentation model on the user's annotated Sogdian pages.

**This phase has two sub-tracks:**

#### 4a — Segmentation model training (NEW, this plan)
**Data:** User's annotated pages (eScriptorium PAGE XML with region polygons + baselines + transcriptions).

**Method (RunPod GPU):**
```bash
# Convert legacy model if needed (not required for seg, only for rec)
# Segmentation trains from default blla.mlmodel as starting point

ketos --config seg_experiment.yml segtrain
```

`seg_experiment.yml`:
```yaml
precision: 32-true
device: cuda:0
segtrain:
  training_data: [train.lst]
  evaluation_data: [val.lst]
  format_type: xml
  checkpoint_path: seg_checkpoints
  weights_format: safetensors
  topline: false
  augment: true
  line_class_mapping:
    - ['*', 3]
    - ['DefaultLine', 3]
    - ['HeadingLine', 4]
    - ['InterlinearLine', 5]
  region_class_mapping:
    - ['*', 8]
    - ['MainZone', 8]
    - ['MarginTextZone', 9]
    - ['NumberingZone', 10]
    - ['DamageZone', 11]
    - ['GraphicZone', 12]
    - ['DigitizationArtefactZone', 13]
    - ['CustomZone', 14]
  quit: early
  epochs: 50
  lrate: 2e-4
  schedule: cosine
  warmup: 200
```

**Files:**
- New: `msocr/training/seg_experiment.yml` (template)
- Modify: `msocr/training/runpod_runner.py` (add segtrain job type alongside existing ketos train)
- Modify: `msocr/training/ketos_trainer.py` (add `segtrain` subcommand wrapper)
- Output: `models/kraken/sogdian_seg.safetensors`

**Validate (gate — this is the critical gate):**
- **Line IoU ≥ 0.85** on held-out Sogdian validation set.
- Region typing accuracy: MainTextZone / MarginalTextZone / NumberingZone correctly classified on ≥ 90% of validation regions.
- If IoU < 0.85: iterate (more data, more epochs, different augment). If still failing after 3 iterations: fall back to default `blla.mlmodel` + manual post-filter (documented degradation, not silent).

#### 4b — Recognition model fine-tuning (PARALLEL, covered by existing plan)
**Reference:** `docs/plans/2026-06-17-msocr-training-pipeline-implementation.md` Phase 2.
- `ketos convert -o sogdian_manuscript.safetensors models/kraken/sogdian_manuscript.mlmodel`
- `ketos --config rec_experiment.yml train` with `--load sogdian_manuscript.safetensors --resize new`
- Output: `models/kraken/sogdian_htr.safetensors`

This track is **not duplicated** in this plan. It runs in parallel with 4a. Both must complete before Phase 5.

**Ponytail note:** YAML config, not custom training loops. RunPod runner extended, not rewritten.

---

### Phase 5 — Recognition Inference
**Goal:** Run fine-tuned segmentation + recognition on new Turfan photos.

**Method (Python API — lib-4 confirmed this is cleanest for JSON output):**
```python
from kraken.tasks import SegmentationTaskModel, RecognitionTaskModel
from kraken.configs import SegmentationInferenceConfig, RecognitionInferenceConfig

seg_model = SegmentationTaskModel.load_model('models/kraken/sogdian_seg.safetensors')
rec_model = RecognitionTaskModel.load_model('models/kraken/sogdian_htr.safetensors')

for fragment in fragments:  # from Phase 1-3
    im = fragment.color_image  # deskewed, NOT binarized
    seg = seg_model.predict(im, SegmentationInferenceConfig())
    for record in rec_model.predict(im, seg, RecognitionInferenceConfig()):
        yield {
            'fragment_id': fragment.id,
            'text': record.prediction,
            'confidence': record.confidences,
            'baseline': record.line.baseline,
            'boundary': record.line.boundary,
            'region_type': record.line.tags.get('type'),
        }
```

**Files:**
- Modify: `msocr/segmentation/kraken_blla.py` (consume `.safetensors`, use `SegmentationTaskModel`)
- Modify: `msocr/models/inference.py` (consume `seg` object, emit per-line JSON)
- New: `msocr/output/page_xml.py` (per-fragment PAGE XML writer with `FRAGMENT_ID` metadata)

**Validate (gate):**
- End-to-end run on `page-19.png`.
- Output: `tmp/phase5_output.json` (per-line records) + `tmp/phase5_output.xml` (PAGE XML per fragment).

---

### Phase 6 — Verification
**Goal:** Quantitative comparison vs current union-bbox pipeline.

**Metrics:**
1. **Line IoU** between predicted baselines/boundaries and held-out ground truth.
2. **CER** between recognized text and ground truth transcription.
3. **Region typing accuracy** (new metric, not in current pipeline).

**Method:**
- Held-out Sogdian validation set (not used in Phase 4 training).
- Compare: (a) current pipeline (union-bbox + default BLLA), (b) new pipeline (CC isolation + fine-tuned seg + fine-tuned HTR).
- Gate: new pipeline CER ≤ current pipeline CER, and Line IoU ≥ 0.85.

**Files:**
- New: `msocr/evaluation/segmentation_metrics.py` (Line IoU + region typing accuracy)
- Extend: `msocr/evaluation/metrics.py` (CER/WER — already exists)

**Validate (gate):**
- Report: `tmp/phase6_verification.md` with IoU, CER, region accuracy for both pipelines.
- If new pipeline fails CER gate: diagnose (segmentation IoU? recognition? both?) and iterate.

---

## 6. Out of Scope

- **Cross-fragment reading order** — philological problem, manual scholarly task. Pipeline outputs per-fragment PAGE XML with `FRAGMENT_ID`; ordering is post-hoc.
- **eScriptorium** — user hard constraint (D3 in design doc). Custom annotation UI inside msocr, covered by existing `2026-06-17` plan.
- **PERO-OCR** — no public training code. Future upgrade only.
- **dfine_kraken** — optional future region detector. Not primary.
- **Searchable PDF output** — explicitly not supported per AGENTS.md.
- **Tesseract/OCRmyPDF/printed-OCR/HAR** — out of scope per AGENTS.md.

---

## 7. SegmOnto Label Vocabulary (from lib-5)

**Official SegmOnto types** (https://segmonto.github.io/):

Zones (15): `MainZone`, `MarginTextZone`, `NumberingZone`, `DropCapitalZone`, `GraphicZone`, `MusicZone`, `QuireMarksZone`, `RunningTitleZone`, `TitlePageZone`, `TableZone`, `SealZone`, `StampZone`, `DamageZone`, `DigitizationArtefactZone`, `CustomZone`

Lines (6): `DefaultLine`, `HeadingLine`, `DropCapitalLine`, `InterlinearLine`, `MusicLine`, `CustomLine`

**Syntax** (with optional `:subtype` and `#number`):
```xml
<TextRegion id="r_1" custom="structure {type:MainZone:column#1;}">
<TextLine id="l_1" custom="structure {type:DefaultLine;}">
```

**For Sogdian Turfan fragments** (selected set):
- `MainZone` — primary text block on the fragment
- `MarginTextZone` — marginal annotations, glosses, additions
- `NumberingZone` — page/folio numbers
- `DamageZone` — Turfan fragments are often damaged (holes, stains, tears); subtypes `:hole`, `:stained`, `:charred`, `:foxed`, `:discoloured`
- `GraphicZone` — illuminations, decorative elements, line-fillers; subtype `:illumination`
- `DigitizationArtefactZone` — ruler and Munsell color card in Turfan photos; subtypes `:ruler`, `:colorTarget`
- `CustomZone` — escape hatch for Sogdian-specific features

Not needed: `MusicZone`, `TableZone`, `TitlePageZone`, `QuireMarksZone`, `RunningTitleZone`, `SealZone`, `StampZone`, `DropCapitalZone`.

**Kraken parsing** (lib-5 confirmed): reads `custom` attribute on `<TextRegion>`/`<TextLine>`, extracts `type:` value. Regions without `custom` default to `'text'` (filterable via `--valid-regions`). Flat ontology — no hierarchy, each `type:subtype` string is a distinct class label.

**Index constraint** (Kraken commit 93e0176): baseline and region label indices must be ≥2 (0–1 reserved for `_start_separator`/`_end_separator` aux classes) **and occupy disjoint ranges** — overlapping indices would merge lines and regions into the same output channel. Plan's YAML uses baselines 3–5, regions 8–14 — disjoint, correct.

## 8. Annotation UI (from lib-5)

**Hard constraint**: user refuses eScriptorium (design doc D3). Custom annotation UI inside msocr (Gradio-based per existing `2026-06-17` plan).

**Problem**: Gradio has **no native polygon/polyline drawing**. `gr.Image` is display-only, `gr.AnnotatedImage` is display-only, `gr.ImageEditor` is brush-only. No existing Gradio custom component supports both polygon drawing AND polyline (baseline) drawing. `gradio-polygonannotator` is a viewer/selector, not a freehand drawing tool.

**Shortest path** (lib-5 recommendation): `gr.HTML` + **Fabric.js** (CDN, no build step).

Architecture:
1. `gr.HTML` embeds `<canvas>` + Fabric.js from CDN.
2. Fabric.js handles: polygon drawing (click to add vertices, dbl-click to close), polyline drawing for baselines, vertex editing (drag), undo/redo, zoom/pan, object selection/deletion, serialize to JSON.
3. Two modes: **region mode** (draw polygon → assign SegmOnto zone type from dropdown) and **baseline mode** (draw polyline → assign SegmOnto line type → enter transcription text).
4. "Save" button → Gradio event sends JSON to Python backend → backend writes PAGE XML.
5. Load existing: Python reads PAGE XML → converts to Fabric.js JSON → sends to frontend.

Frontend → Backend JSON:
```json
{
  "image": "path/to/page.png",
  "image_width": 2480,
  "image_height": 3508,
  "regions": [
    {
      "id": "r_1",
      "type": "MainZone",
      "subtype": "column",
      "number": 1,
      "points": [[100, 200], [800, 200], [800, 1200], [100, 1200]],
      "lines": [
        {
          "id": "l_1",
          "type": "DefaultLine",
          "baseline": [[120, 250], [780, 250]],
          "text": "transcription text here"
        }
      ]
    }
  ]
}
```

Backend → PAGE XML:
```xml
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="page.png" imageWidth="2480" imageHeight="3508">
    <TextRegion id="r_1" custom="structure {type:MainZone:column#1;}">
      <Coords points="100,200 800,200 800,1200 100,1200"/>
      <TextLine id="l_1" custom="structure {type:DefaultLine;}">
        <Coords points="120,230 780,230 780,270 120,270"/>
        <Baseline points="120,250 780,250"/>
        <TextEquiv><Unicode>transcription text here</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

`<Coords>` on `<TextLine>` (line bounding polygon) can be auto-computed from baseline + estimated line height if not manually drawn.

**What was skipped**: full Gradio custom component (Svelte + npm build chain) — `gr.HTML` + Fabric.js CDN is one file, no build. OpenLayers/Leaflet — wrong abstraction (pixel coords, not geographic). `gradio-polygonannotator` — viewer only.

**Add when**: state management becomes unwieldy (multi-page, collaborative editing) → graduate to a proper Svelte Gradio custom component.

## 9. Open Questions (need user confirmation before implementation)

1. **Annotation data format confirmation** — user said "XML or other format suit for Kraken fine-tuning." Is the current annotation output eScriptorium PAGE XML with SegmOnto labels (MainTextZone/MarginalTextZone/NumberingZone)? If not, what format? This determines whether Phase 4a data is ready as-is or needs a conversion step.

2. **CER target X** — the gate is "CER ≤ X vs current union-bbox pipeline." What is X? (Suggested: X = current pipeline CER, i.e., "no worse than today" as the minimum bar, with a stretch goal of CER ≤ 0.15.)

3. **Annotation UI status** — the existing `2026-06-17` plan covers a custom annotation UI. Is that UI implemented and producing PAGE XML already, or is annotation currently happening in a different tool? This affects whether Phase 4a can start immediately or waits on the annotation UI.

4. **Held-out validation set** — how many pages should be held out for Phase 6 verification? (Suggested: 10% of annotated pages, minimum 5 pages.)

5. **Phase execution order** — proceed Phase 1 → 2 → 3 → 4 → 5 → 6 sequentially with gates, or parallelize Phase 4a (seg training) with Phase 4b (rec training, already in existing plan)? (Suggested: parallelize 4a and 4b since they share no data and both run on RunPod.)

---

## 8. What Was Skipped (Ponytail)

- No SAM2, YALTAi, Swin-UNet, PERO — verified wrong task / archived / no code / no training code.
- No custom binarization class — Sauvola is stdlib, nlbin is one import.
- No custom deskew algorithm — extends existing `correct_skew`.
- No custom training loop — Kraken 7 YAML config + `ketos segtrain`.
- No duplicate of HTR fine-tuning infrastructure — references existing `2026-06-17` plan.
- No cross-fragment reading order — out of scope (philological).
- No new abstraction for region types — SegmOnto labels via `region_class_mapping`.

**Add when:** Line IoU consistently < 0.85 after 3 training iterations → consider dfine_kraken as region detector upgrade. CER consistently > 0.20 → consider PERO-OCR inference (if license acceptable) as recognition upgrade, accepting the no-training-code constraint.