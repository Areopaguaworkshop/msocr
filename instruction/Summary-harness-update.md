# Manuscript OCR System for Church Fathers (`msocr`)

## 1. Purpose and Scope

- Build and implement an OCR/HTR system for historical Church Fathers corpora.
- Support both printed OCR and handwritten HTR, with strict route separation.
- Keep current CLI-compatible flow: `preprocess -> annotate -> train -> ocr`.
- Ground-truth annotation is a first-class pipeline stage, not an external dependency.
- Implementation rollout: Syriac printed OCR is the first automated pipeline target.
  Greek and Latin printed OCR follows once the Syriac pipeline is validated end-to-end.

## 1.1 Implementation Policy

- Code implementation is allowed and expected in this phase.
- All project code must be created under `msocr/` with clear module structure.
- Use `uv` for environment management, dependency handling, and command execution.
- No feature is considered implemented until a code path exists and a CER/WER result
  is recorded against a frozen benchmark manifest.

---

## 2. Language Registry

All supported languages are registered here with their script properties.
These properties are the authoritative source for the annotation tool, the CLI router,
and the Harness pipeline parameter set.

| Language        | Direction | Unicode Block(s)                    | Web Font                  | Variants / Scripts              | OCR Engine (printed)       | HTR Path              |
|-----------------|-----------|-------------------------------------|---------------------------|---------------------------------|----------------------------|-----------------------|
| Syriac          | RTL       | U+0700–U+074F                       | Noto Sans Syriac          | Estrangela, Serto, East Syriac  | Tesseract `syr` + finetune | Kraken (Transkribus bridge) |
| Sogdian         | RTL       | U+10F30–U+10F6F                     | Noto Sans Sogdian         | Formal (Sutra), Cursive         | Kraken (custom)            | Kraken (custom)       |
| Old Sogdian     | RTL       | U+10F00–U+10F2F                     | Noto Sans Old Sogdian     | Ancient Letters                 | Kraken (custom)            | Kraken (custom)       |
| Old Turkish     | RTL       | U+10C00–U+10C4F (Old Uyghur)        | Noto Sans Old Turkic      | Old Uyghur script               | Kraken (`configs/old_turkish_config.yaml`) | Kraken |
| Greek           | LTR       | U+0370–U+03FF, U+1F00–U+1FFF        | GFS Didot / Noto Serif    | Polytonic, Minuscule, Uncial    | Kraken primary + fallback  | Kraken                |
| Latin           | LTR       | U+0000–U+007F, U+0100–U+024F        | Junicode / EB Garamond    | Caroline, Insular, Gothic       | Kraken CATMuS-Print Large  | Kraken                |
| Coptic          | LTR       | U+2C80–U+2CFF (+ U+03E2–U+03EF)     | Noto Sans Coptic / Antinoou | Sahidic, Bohairic             | Tesseract `cop`            | Kraken (custom)       |
| Armenian        | LTR       | U+0530–U+058F                       | Noto Sans Armenian        | Erkat'agir, Bolorgir, Notrgir   | Tesseract `hye-calfa-n`    | Kraken                |
| Geez / Ethiopic | LTR       | U+1200–U+137F                       | Noto Sans Ethiopic        | Ge'ez script                    | Tesseract `gez`            | Kraken                |

**Notes:**
- Sogdian and Old Sogdian are Supplementary Multilingual Plane scripts (above U+FFFF).
  Browser `input` fields and font stacks must handle surrogate pairs correctly.
- Old Turkish in `msocr` uses the Old Uyghur script block, not the Old Turkic runic block (U+10C00).
  The existing `configs/old_turkish_config.yaml` must be audited to confirm the correct Unicode range.
- Coptic requires both the dedicated Coptic block (U+2C80) and the legacy Greek block
  entries (U+03E2–U+03EF) for complete scholarly coverage.
- Polytonic Greek requires combining characters from Greek Extended (U+1F00–U+1FFF);
  the annotation input field must not strip or decompose these.

---

## 3. Annotation API (`msocr/service/annotation_api.py`)

The annotation API is a browser-accessible FastAPI extension that provides
ground-truth collection for all supported languages. It is language-agnostic:
direction, font, and Unicode range are session parameters, not hardcoded values.

### Design principles

- A session normalises all three ingestion paths into one internal contract:
  image on disk + pixel-domain line coordinates + language metadata.
- Segmentation always runs before annotation. The scholar annotates line crops,
  not raw page images.
- The tool exports formats that `ketos train` consumes directly, with no manual
  post-processing required.

### Three ingestion paths (all supported, same internal contract)

| Path | Trigger | Notes |
|------|---------|-------|
| Browser upload | Scholar drops TIFF/JPEG/PNG/PDF via browser | Max recommended 300 dpi; PDF converted page-by-page |
| Local file path | Path relative to `msocr/` project root or absolute | Used in automated pipeline pre-annotation step |
| IIIF manifest URL | IIIF Presentation API v2 or v3 | Canvases loaded as ordered page sequence |

### Segmentation fallback chain (runs automatically on session creation)

```
Level 1 — kraken blla (baseline segmenter)
  → always attempted first
  → outputs baseline + boundary polygon per line in pixel coordinates

Level 2 — kraken pageseg (legacy bounding-box segmenter)
  → triggered if blla returns fewer than 3 lines OR raises an exception
  → emits warning: needs_manual_review: true on session

Level 3 — manual correction in browser
  → always available regardless of level 1/2 result
  → scholar can skip bad segments or (v0.2) redraw boundaries
```

### API endpoints

```
POST   /api/sessions
  Body: { language, script_variant, ingestion_path, source }
  Returns: { session_id, page_count, line_count, segmentation_engine, lines[] }

GET    /api/sessions/{id}
  Returns: full session state including all line crops, coordinates, annotations to date

GET    /api/sessions/{id}/line/{n}/image
  Returns: cropped line image (JPEG) derived from pixel-domain bounding box

POST   /api/sessions/{id}/save
  Body: { annotations: [{ line_id, transcript, skip }] }
  Returns: updated session state

GET    /api/sessions/{id}/export?format={alto|page|tsv}
  Returns: downloadable file ready for `ketos train`
```

### Export format requirements

All exports must carry language and script variant as metadata:

- **ALTO XML**: `LANG` attribute on `<TextBlock>`, `TAGREFS` referencing a
  `<Tags><OtherTag>` element declaring the script variant. `TextLine` elements
  carry pixel-domain `HPOS`, `VPOS`, `WIDTH`, `HEIGHT`, `BASELINE`.
- **PAGE XML**: `<TextRegion>` carries a `custom` attribute with
  `language:{ value:<lang>; }` and `script:{ value:<variant>; }`.
  `<TextLine>` elements carry `<Baseline>` and `<Coords>` in pixel space.
- **TSV**: `<image_path>\t<transcript>` one pair per line. Image paths are
  relative to `msocr/data/sessions/{id}/crops/`.

### Session persistence

Sessions are stored under `msocr/data/sessions/{id}/`:
```
{id}/
  session.json          ← metadata: language, script, segmentation engine, line list
  page.tif              ← normalised page image
  crops/
    line_001.jpg
    line_002.jpg
    ...
  annotations.json      ← transcript per line_id, updated on each /save call
```

A scholar can close the browser and resume. Sessions do not expire automatically.

### Language-specific rendering requirements

- **RTL languages** (Syriac, Sogdian, Old Sogdian, Old Turkish): `input` element
  must carry `dir="rtl"` and `lang="{iso}"`. CSS `text-align: right` must be set
  explicitly; do not rely on the browser bidi algorithm alone.
- **Supplementary plane scripts** (Sogdian U+10F30+, Old Sogdian U+10F00+,
  Old Turkish U+10C00+): fonts must be served locally or from a trusted CDN.
  Google Fonts serves Noto Sans Sogdian and Noto Sans Old Sogdian.
  Do not assume system font fallback for these scripts.
- **Polytonic Greek**: the input field must preserve combining diacritics
  (U+1F00–U+1FFF). Do not set `spellcheck="true"` — browser spell check
  will corrupt polytonic characters.
- **Coptic**: use Noto Sans Coptic or Antinoou. Do not use generic Greek fonts;
  the letterforms are distinct and scholars will reject incorrect rendering.

---

## 4. Route Separation (Required)

### Printed OCR route
- Preprocessing and model profiles optimised for print.
- Primary metric: `CER`. Acceptance thresholds are per-language and per-variant
  (see Section 5).
- Secondary metric: `WER` tracked but not a gate.

### Handwritten HTR route
- Preprocessing and model profiles optimised for handwriting.
- Primary metric: `CER <= 10%` as a starting gate; tighten per-language as
  fine-tuned models mature.
- For all languages without a strong public HTR model, custom Kraken training
  from Transkribus-exported PAGE/ALTO XML is the required path.

### Routing rules
- `writing_mode` must be declared at session and pipeline design time
  (`printed` or `handwritten`). No runtime auto-detection in production.
- If `writing_mode=auto` is used for exploration, output must include explicit
  model confidence and must emit `needs_manual_review: true`.
- RTL languages (Syriac, Sogdian, Old Sogdian, Old Turkish) require RTL
  normalisation of output text and line ordering. This is not optional.

---

## 5. Benchmark and Evaluation Policy

### CER acceptance thresholds (per language and variant)

| Language        | Variant / Script    | Route    | CER Gate | Notes |
|-----------------|---------------------|----------|----------|-------|
| Syriac          | Estrangela          | printed  | ≤ 5%     | Achievable with stock Tesseract `syr` |
| Syriac          | Serto               | printed  | ≤ 10%    | Requires fine-tuning; tighten after corpus |
| Syriac          | East Syriac         | printed  | ≤ 10%    | Hardest baseline; likely needs dedicated model |
| Syriac          | all variants        | HTR      | ≤ 10%    | Starting gate only |
| Greek           | Polytonic           | printed  | ≤ 5%     | Kraken + CATMuS-Print |
| Latin           | Historical          | printed  | ≤ 5%     | Kraken CATMuS-Print Large |
| Coptic          | Sahidic/Bohairic    | printed  | ≤ 5%     | Tesseract `cop` |
| Armenian        | all scripts         | printed  | ≤ 5%     | Tesseract `hye-calfa-n` preferred |
| Geez            | Ge'ez script        | printed  | ≤ 5%     | Tesseract `gez` |
| Sogdian         | all styles          | printed  | ≤ 10%    | No strong public baseline; custom Kraken |
| Old Turkish     | Old Uyghur          | printed  | ≤ 10%    | Existing Kraken config |

**All thresholds are start-of-project gates, not permanent targets.**
Tighten after measuring actual baseline CER on real corpus samples.
The flat `CER <= 5%` claim in earlier documentation is not valid for
Serto, East Syriac, Sogdian, or Old Turkish without fine-tuning.

### Evaluation requirements
- Both CER and WER are required outputs from every training run.
- Results must be written to `metrics.json` with fields:
  `cer`, `wer`, `language`, `script_variant`, `writing_mode`,
  `model_version`, `pipeline_run_id`, `manifest_id`.
- Split policy: predefined frozen manifests grouped by `manuscript_id`.
  Manifests are committed to the repository under `data/manifests/`.
  No dynamic train/test splitting allowed in gated runs.

---

## 6. Training Data Policy

- Minimum useful fine-tuning: 500–1,000 aligned line pairs.
- Strong baseline: 2,000–5,000 aligned line pairs.
- All ground truth is collected via the annotation API (`/api/sessions`).
- Line-level image/transcript pairs are the primary training artifact.
- Page-level annotations are kept for segmentation model training only.
- Data versioning: all training manifests are DVC-tracked. No training run
  may use an unversioned data split.

---

## 7. CI/CD Pipeline — Harness + RunPod

### Platform constraints (Harness Free / Developer tier)

- Harness Cloud runners are not available on Free tier for non-business email accounts.
- All CI/CD execution uses a **self-hosted Harness Delegate** running on a local
  dev machine. The delegate has no build minute limits.
- Harness Artifact Registry (HAR) is used for model storage. Generic artifact type
  stores `.traineddata` (Tesseract) and `.mlmodel` (Kraken) files.
- RunPod provides GPU compute for training. Integration is via RunPod REST API
  called from a `Run` step in the Harness pipeline.

### Pipeline stage sequence

```
1. Trigger
   Git push to data/syriac/** or data/annotation/** path filter
   OR manual dispatch with language + variant parameters

2. Validate
   - Check annotation session export: line count >= 500, XML schema valid
   - Confirm frozen manifest exists in data/manifests/
   - Assert script_variant declared in session metadata

3. Build training image
   - Docker image: ubuntu:22.04 + Tesseract 5.x + tesstrain + tessdata_best/syr.traineddata (SHA-pinned)
   - Push to HAR Docker registry

4. RunPod: submit training job
   - POST to RunPod API with image ref, GPU tier (RTX 4090 default), environment variables
   - Poll RunPod job status OR receive webhook callback on completion

5. Evaluate
   - Run ketos test (Kraken) or tesseract eval (Tesseract) on frozen benchmark manifest
   - Write metrics.json with CER, WER, language, script_variant, model_version, run_id

6. Policy gate
   - Read metrics.json
   - Apply per-variant CER threshold from Section 5
   - PASS: continue to register step
   - FAIL: emit needs_manual_review: true, block deployment, notify

7. Register artifact
   - Push model file to HAR as generic artifact
   - Artifact name pattern: {lang}-{variant}-v{pipeline.sequenceId}
   - Attach metrics.json as artifact sidecar
   - Tag: staging (automatic), production (manual approval gate)

8. Deploy (on production tag only)
   - Harness CD pulls model from HAR on FastAPI service startup
   - Gradio demo updated to point to new model tag
```

### RunPod GPU tier selection

| Training scenario | Recommended tier | Notes |
|-------------------|-----------------|-------|
| Tesseract finetune (tesstrain) | RTX 3090 / 4090 (24 GB) | LSTM fine-tuning is not memory-intensive |
| Kraken ketos train (small corpus) | RTX 3090 / 4090 (24 GB) | Adequate for < 5,000 lines |
| Kraken ketos train (large corpus) | A100 40 GB | Only if corpus > 20,000 lines |

### Artifact versioning scheme (HAR)

```
Registry name:  msocr-models
Artifact type:  generic

Name pattern:   {lang}-{script_variant}-{writing_mode}-v{sequenceId}
Examples:
  syr-estrangela-printed-v14
  syr-serto-printed-v15
  grc-polytonic-printed-v3
  sog-formal-htr-v2

Sidecar files (attached to each version):
  metrics.json     ← CER, WER, manifest_id, run_id
  config.yaml      ← training hyperparameters
  Dockerfile.sha   ← SHA of training image used
```

### Delegate installation (local dev machine)

- Install Harness Delegate (Docker or Kubernetes mode) on dev machine.
- Register delegate under project: `msocr` / org: `{your-org}`.
- Delegate tags: `local`, `gpu-proxy` (the delegate calls RunPod, not the GPU itself).
- Delegate does not need GPU; it only submits API calls and polls results.
- Store RunPod API key as a Harness secret: `RUNPOD_API_KEY`.

### Pipeline YAML location

```
pipeline/harness/
  syriac_printed_train.yaml     ← Syriac printed OCR pipeline (first to build)
  greek_printed_train.yaml      ← planned
  latin_printed_train.yaml      ← planned
  htr_generic_train.yaml        ← planned (shared HTR pipeline, language-parameterised)
```

---

## 8. Integration Mapping to Existing `msocr` Modules

### Implemented components
- CLI orchestration: `msocr/cli.py`
- Dataset metadata and storage: `msocr/data/manager.py`
- Annotation bridge (Label Studio / CVAT / ALTO / PAGE): `msocr/data/annotation.py`
- Preprocessing pipeline: `msocr/preprocessing/pipeline.py`
- Inference wrapper: `msocr/models/inference.py`
- Kraken training wrapper: `msocr/training/ketos_trainer.py`
- FastAPI backend service: `msocr/service/api.py`
- Gradio demo service: `msocr/service/gradio_demo.py`
- Sogdian config: `configs/sogdian_config.yaml`
- Old Turkish config: `configs/old_turkish_config.yaml`

### New components to build (in priority order)

| Component | Path | Dependency |
|-----------|------|-----------|
| Annotation API | `msocr/service/annotation_api.py` | Requires: Kraken blla in service layer |
| Annotation session store | `msocr/data/session_manager.py` | Requires: annotation API |
| Frozen manifest manager | `msocr/data/manifest.py` | Requires: DVC setup |
| CER/WER evaluation runner | `msocr/evaluation/metrics.py` | Requires: manifest manager |
| RunPod job client | `msocr/pipeline/runpod_client.py` | Requires: RunPod API key secret |
| HAR artifact client | `msocr/pipeline/har_client.py` | Requires: Harness API token |
| Language router | `msocr/models/router.py` | Requires: language registry |
| Training container spec | `docker/train/Dockerfile` | Blocks RunPod integration |

### Planned additions (after above are complete)
- Script/language + mode classifier layer
- Lexicon/morphology post-correction module
- Expanded language profiles (per language in registry)
- eScriptorium export adapter

---

## 9. Deliverables

### Documentation (this baseline)
- `instruction/summary.md` (this file)
- `instruction/agent.patristic.ocr.model.md`
- `instruction/eval.md`
- `instruction/code_review.md`
- `instruction/annotation_api.md`

### Pipeline definitions
- `pipeline/harness/syriac_printed_train.yaml`
- `pipeline/harness/htr_generic_train.yaml`

### Skill and registry YAMLs
- `skill/` — per-language skill definitions
- `source_registry/` — corpus source registry
- `output/` — state update YAMLs

---

## 10. Maintenance Policy

- This file is the authoritative planning baseline for the current cycle.
- Later updates must be incremental and tied to actual measured progress.
- Do not promote planned items to implemented until a code path exists
  AND a CER/WER result is recorded against a frozen manifest.
- When a CER threshold is revised based on real measurements, update the
  table in Section 5 and record the reason in git commit message.
- Keep language registry (Section 2) synchronised with the annotation API
  language configuration object in `msocr/service/annotation_api.py`.

---

## 11. Current Execution Notes

### Syriac printed — first automated pipeline target
- Priority: Syriac printed, mixed corpus (Estrangela + Serto + East Syriac).
- Ground truth: starting from zero. Annotation API must be built and deployed
  before any training run can be scheduled.
- Baseline benchmark (no training): run Tesseract `syr` on sample Estrangela
  pages to confirm stock CER before any fine-tuning investment.
- Fine-tuning: tesstrain on tessdata_best/syr.traineddata checkpoint.
  Separate fine-tuned models per script variant if mixed corpus CER is
  above threshold for any variant.
- Harness pipeline: `pipeline/harness/syriac_printed_train.yaml` (to be authored).
- RunPod GPU: RTX 4090 (24 GB) is adequate for tesstrain on 500–5,000 line corpus.

### Syriac handwritten
- Transkribus platform for first-pass HTR; export PAGE/ALTO XML with corrected text.
- Use exported XML + images as Kraken training data.
- Custom Kraken HTR model is the target; Transkribus is the bootstrap, not the end state.

### Greek and Latin printed
- Greek: Kraken primary with CATMuS-compatible model; Tesseract fallback.
- Latin: Kraken CATMuS-Print Large; Tesseract fallback.
- Both routes described as implemented in `Start-here.md`; validate with real CER
  measurement before marking as pipeline-ready.

### Coptic printed
- Tesseract `cop` is the primary route.
- OCRopus/Ocropy fallback is deactivated.

### Coptic handwritten
- Custom Kraken HTR training required; Transkribus workflow for dataset generation.
- Tesseract kept as backup training option for specific corpora.

### Armenian printed
- Tesseract `hye-calfa-n` preferred; standard `hye` as fallback.
- `potmind/armenian-ocr` route dropped.

### Armenian handwritten
- Kraken training from PAGE/ALTO XML exported from eScriptorium or Transkribus.
- Image cleanup (noise removal, deskew, crop) must precede layout analysis.

### Sogdian (printed and HTR)
- No strong public model. Custom Kraken training required for both routes.
- Existing `configs/sogdian_config.yaml` covers the training skeleton;
  ground truth collection is the blocker.
- Script renders RTL; use Noto Sans Sogdian for annotation tool and output preview.

### Old Turkish (Old Uyghur script)
- Existing `configs/old_turkish_config.yaml`.
- Confirm Unicode range used in config matches Old Uyghur block (U+10F70–U+10FAF),
  not the Old Turkic runic block (U+10C00–U+10C4F).
- RTL rendering required in all output handling.

### Geez printed and handwritten
- Printed: Tesseract `gez`.
- HTR: same preparation and training sequence as Armenian handwritten.
