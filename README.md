# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

`msocr` is a focused Sogdian manuscript HTR toolkit. It uses Kraken for local handwritten text recognition, keeps language handling Sogdian-only, and provides small tools for ground-truth preparation, model training (local or remote on RunPod GPU Cloud Pods), inference, an API, and a React annotation UI.

Active scope (remote training): RunPod GPU Cloud Pod submission for `ketos train` fine-tuning via `msocr train-remote`, with a minimal procedural per-style-group orchestrator (one style-group at a time, not a DAG engine).

Removed scope: printed OCR routing, Tesseract/OCRmyPDF fallbacks, benchmark promotion flows, artifact registry publication, and multi-language orchestration.

## Christian Sogdian Fine-tune — Fix A v3 / 0.2 (shipped, current best)

**Final model**: `models/kraken/c2av_finetune_200ep` (15.3MB). Holdout CER **71.43%** (beats Fix A v2's 77.75% by 6.32 points). Path: identical recipe to v2 (`--resize union --freeze-backbone 999999 --augment --warmup 200 --lr 1e-4`) but with `--epochs 200 --lag 25 --min-epochs 8` — the 0.1 confusion-matrix audit showed 83.4% of errors are CTC-alignment deletions (model emits blank instead of a char), not misclassifications, so the lever is *more training*, not a new architecture. Best checkpoint at epoch 173 (val score 0.4043).

| Plate | Role | CER (0.2, 200ep) | CER (v2, 60ep) | CER (first run, new-frozen) |
|---|---|---|---|---|
| c2av12 | holdout (unseen) | **71.43%** | 77.75% | 82.12% |
| c2av11 | validation | **59.57%** | 65.96% | 82.50% |
| c2av01 | training (seen) | **6.08%** | 45.30% | 83.43% |

All three improved; train collapsed from 45% to 6% (severe overfit — gap ~9.8×). Val/holdout gains are modest because the binding constraint is now per-plate generalization, not CTC alignment. Reports: `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune_200ep.{json,md}` (holdout) + `scripts/triangulate_eval.py` for the full triangulation.

### What worked (incremental over v2)

- **0.1 confusion-matrix audit** (`scripts/confusion_audit.py`, `reports/confusion_c2av12_analysis.md`): 332 errors / 427 chars on c2av12 holdout. 277 deletions (83.4%), 51 substitutions (15.4%), 4 insertions (1.2%). 87.3% of errors on shared classes (already in sophro base). Diagnosis: CTC alignment collapse, not classifier failure → lever is more training, not `--append` (which was skipped — would have lost shared weights and made it worse).
- **`--epochs 200 --lag 25`**: gave CTC alignment 3.3× more epochs to converge. Best checkpoint moved from epoch 49 (3.2, 60ep run) to epoch 173 (0.2, 200ep run). val_acc still climbing at epoch 60 was the underfitting signal that motivated this run.

### What did not work (v3 levers tried and dropped)

- **3.1 Lexicon post-corrector** (`scripts/lexicon_postcorrect.py` + Sims-Williams 2021 Excel corpus, 7,387 unique words): increases CER from 79.90% to 85.57% on holdout. Lexicon can only fix substitutions (15.4% of errors); at CER ~80% broken tokens are closer to truth than any edit-distance-2 lexicon match. Revisit only if CER drops below ~50%. See `docs/fixa-v3-3.1-lexicon-negative-result.md`.
- **3.2 Beam search**: `beam_decoder` removed in kraken 7.0b1. Porting effort high, expected gain 0–2% (often zero). Dropped.
- **1.1 `--append`**: would discard the working shared-class weights. 0.1 audit confirmed 87.3% of errors are on shared classes — `--append` makes it strictly worse. Skipped.
- **2.1 SogdianOCR CycleGAN**: does not exist publicly. Synthetic data path requires building from `kraken/linegen.py` + Noto Sans Syriac Eastern (high effort, deferred).

### Post-processor

`scripts/syriac_to_latin.py` (~60 lines, stdlib only) reverses the Syriac-script output back to Latin transliteration. Deterministic 1:1 reverse lookup for 22 base consonants + 3 Sogdian letters, with TAW+qushshaya→`t` / TAW+rukkakha→`θ` diacritic handling and a small Sogdian wordform dictionary (`SOFT_DICT`) for hard/soft disambiguation of GAMAL (g/γ) and DALATH (d/δ). Round-trip tested: 23/23 chars recover correctly.

### Next steps to push CER down

1. **Annotate more plates** (target 500–1000+ lines across multiple manuscripts). 148 lines is the binding constraint. Train CER 6% vs val 59.57% = overfit; the model has memorized the 10 train plates and cannot generalize to unseen plates at this data volume. More data is the only path to CER < 10%.
2. **1.2 Oversample rare classes** (one pod, quick): duplicate lines containing rukkakha/zhain/fe 3–5× in the training list. Low effort, modest expected gain.
3. **3.3 Syriac-only fine-tune** (only if 1.2 insufficient): shared-class errors (87.3%) suggest the backbone features don't transfer perfectly to this hand. High effort.
4. **2.1 Synthetic data** (if ceiling reached): `kraken/linegen.py` + Noto Sans Syriac Eastern.

See `docs/fixa-v3-cer-lower.md` for the full v3 plan and `docs/fix-a-v2-syriac-script-finetune.md` for the v2 history.

## Christian Sogdian Fine-tune — First Run (historical, superseded by Fix A v2)

First end-to-end fine-tune from the Sophro Mhiro Syriac base model onto 12 C2AV Berlin plates (Sims-Williams 1985). Establishes the full remote-training pipeline (manifest → polygon enrichment → RunPod submit → SSH train → artifact download → local eval) and produces the first measurable CER numbers for this corpus.

### Setup

- **Base model**: `models/kraken/sophro_mhiro_syriac.safetensors` (Sophro Mhiro Syriac, 36-char Syriac codec, 3,466 lines across 38 manuscripts, 90.99% accuracy on its own domain)
- **Training data**: 12 C2AV plates, 178 lines total, Latin-transliteration Sogdian (24-char codec: `. b c d f g l m n p q r s t w x y z š ž ʾ γ θ`). 0 char overlap with the base model codec.
- **Split**: 10 train plates (148 lines) / 1 val plate p-11 (15 lines) / 1 holdout plate p-12 (19 lines). One honest generalization number on a fully unseen plate.
- **Manifest**: `data/manifests/c2av-finetune.json`. Ground truth at `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml`, plate images at `dataset/christian_sogdian_c2av/plates/p-{01..12}.png`.
- **Fine-tune flags**: `--augment --warmup 200 --lr 1e-4 --freeze-backbone 5000 --resize new --epochs 30 --min-epochs 20 --lag 10 --quit early`. `--resize new` rebuilds the output head for the 24-char Latin codec (the visual CNN backbone is retained from the Syriac base).

### RunPod infrastructure fixes (this run)

Two transient secure-cloud failures hit before training could start; both were runner-side, not training-side:

1. **pip-install timeout** (run 1): setup-cmd timeout bumped 1800s → 3600s. Returns early on success, so the headroom costs nothing.
2. **Capacity-exhaustion silent hang** (run 2, 3): `create_pod` returns a pod ID immediately even when the scheduler can't place the pod on a machine (RTX 3090 secure cloud is capacity-constrained per RunPod supply notice). The pod sits with `desired=None` + `runtime=None` and never exposes an SSH port. Added `PodNeverScheduledError` (detects stuck-in-limbo after 100s) + a 3-retry create+ssh loop in `runpod_runner.py`. Without the fix, a stuck pod hung the runner 600s then crashed; with it, attempts 1 and 2 failed cleanly in ~100s each, attempt 3 landed a GPU and completed the run.

### Results

Training early-stopped at epoch ~20 (best checkpoint epoch 10, val accuracy 0.175). The model was then evaluated on three plates for triangulation:

| Plate | Role | CER | WER | Accuracy |
|---|---|---|---|---|
| c2av01 | training (seen) | 83.43% | 100.0% | 16.57% |
| c2av11 | validation (early-stopping signal) | 82.50% | 100.0% | 17.50% |
| c2av12 | holdout (unseen) | 82.12% | 100.0% | 17.88% |

Reports:
- `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune.{json,md}` (holdout p-12)
- `reports/c2av-eval-val__c2av-syriac-finetune__c2av_finetune.{json,md}` (val p-11)
- `reports/c2av-eval-train__c2av-syriac-finetune__c2av_finetune.{json,md}` (train p-01)

### Diagnosis: 148 lines is the binding constraint, not the freeze schedule

CER is flat at ~82% across training, validation, and holdout plates. The first hypothesis was underfitting from `--freeze-backbone 5000` (the CNN never unfroze before early stopping). A second run with `--freeze-backbone 0` disproved this:

| Plate | Role | CER `--freeze-backbone 5000` | CER `--freeze-backbone 0` |
|---|---|---|---|
| c2av01 | training (seen) | 83.43% | 95.86% |
| c2av11 | validation | 82.50% | 87.50% |
| c2av12 | holdout (unseen) | 82.12% | 91.45% |

Unfreezing the backbone from step 0 made every plate worse and introduced a train>val>holdout gap (overfitting). With 148 lines, neither schedule works: frozen = head can't map Latin transliteration onto frozen Syriac features (flat 82%); unfrozen = backbone catastrophically forgets Syriac features before the new Latin head converges, while overfitting the tiny training set. The binding constraint is data quantity, not the freeze schedule — exactly what the feasibility research predicted.

### What the model is still good for

A CER 82% model is not a publishable recognizer, but it is a working baseline: a human correcting ~18% of characters is faster than transcribing from scratch, and the model improves as more plates are annotated. The infrastructure work is the real deliverable here — the full remote-training pipeline now runs end-to-end without intervention and survives transient RunPod capacity failures.

### Next steps to push CER down

1. **Annotate more plates** (target 500+ lines across multiple manuscripts). 148 lines is below the viable range for fine-tuning regardless of freeze schedule; research suggests 500-1000+ lines for CER < 10%. This is the single binding constraint — flag tuning is exhausted.
2. **Leave-one-plate-out CV** across all 12 plates for a more principled evaluation than a single 19-line holdout.
3. Once more data exists, re-tune `--freeze-backbone` (try 200-1000 for a partial warmup) and `--lr` (try 5e-5 to avoid catastrophic forgetting).

See `docs/kraken-178-lines-feasibility.md` for the feasibility research that predicted 20-40% CER (the run underperformed because 148 lines is below the viable range, not because the freeze schedule was wrong).

## Recent Developments

Active work has shifted toward **Christian Sogdian** manuscripts (Sogdian language written in the East Syriac script, Unicode block `U+0710`) sourced from the Sims-Williams 1985 C2AV Berlin plates. Recent changes:

- **Christian Sogdian manifests** under `data/manifests/`: `c2av-finetune.json` is the active 10/1/1 split, `lopo/` contains leave-one-plate-out variants, and the `*-smoke.json` manifests only exercise the pipeline. Style group `c2av-syriac-finetune` targets the Syriac base model at `models/kraken/sophro_mhiro_syriac.safetensors`.
- **Fragment-handling CLI pipeline** for damaged/fragmentary manuscripts: `msocr isolate-fragments` (Sauvola + connected components + DBSCAN), `binarize-fragments` (per-fragment Sauvola/NLbin with bleed-through detection), `deskew-fragments` (Hough deskew on binary masks), and `extract-lines` (exact-count row-band cropping with optional ROI / row-centers). See `docs/kraken-fragmentary-manuscripts.md`.
- **Annotation UI rewrite**: the browser UI moved from HTMX/Alpine to a React SPA (OpenSeadragon viewer + Phosphor icons, Sogdian Latin-transliteration palette). The SPA bundle lives in `frontend/dist/` and is served by the annotation API. Build it with `cd frontend && npm install && npm run build`. `msocr annotate`, `annotation-api`, and `demo` all refuse to start if `frontend/dist/index.html` is missing.
- **Annotation UI Tier 1 (eScriptorium-parity)**: per-point baseline vertex delete (Ctrl+Del), reading-direction invert (I), join lines (J), explicit line↔region link/unlink (Y/U) with orphan-line indicator, reading-order auto-sort (Shift+L) with order badges, `?` help cheatsheet, plain-text side panel (Ctrl+5), and line-type dropdown. `tsc` + `vite build` pass. See `docs/fix-a-v9-annotation-plan.md` §Tier 1 for the full list and the deliberate simplifications (`ponytail:` comments in `AnnotateEditor.tsx`).
- **`msocr demo`** now launches the annotation API + React SPA on port 8001 (default). The legacy Gradio demo is dead code; `--share` is a silent no-op.
- **`--no-crop-manuscript-area`** on `annotate` and `annotation-api` disables automatic manuscript-area cropping before line segmentation.
- **RunPod runner hardened**: discovers the public SSH host:port from `runtime.ports` (RunPod exposes SSH on a random public port), relies on account-level SSH key injection (no per-pod env var), reduced default pod disk to 50 GB.
- **Orchestrator polygon enrichment**: kraken 7.x requires `<Coords>` per `<TextLine>`, but the annotation exporter only emits `<Baseline>`. The orchestrator now computes polygonal `<Coords>` via `calculate_polygonal_environment` before upload, and applies an idempotent in-process patch for a kraken 7.0.2 checkpoint-save crash (`self.net is None`).
- **Evaluation harness**: parses kraken 7.0.2's percentage-format Character/Word Accuracy output (in addition to legacy `CER:`/`WER:` labels) and enriches baseline-only holdout XMLs into temp polygon XMLs before `ketos test`.
- **New docs**: `docs/kraken-fragmentary-manuscripts.md`, `docs/kraken-training-data-research.md`, `docs/multi-script-htr-research.md`, `docs/runpod-gpu-research.md`, `docs/codeql-scan.md`, `docs/semgrep-scan.md`.

## Supported Language

| Code | Alias | Direction | Runtime font | Default model |
|---|---|---|---|---|
| `sogdian` | `old_sogdian` | RTL | Noto Sans Sogdian | `models/kraken/sogdian_manuscript.mlmodel` |

The registry stays Sogdian-only, but manifests carry a `script_block` (`U+10F30` Sogdian or `U+0710` Syriac) plus an optional `script_variant` (e.g. `christian-syriac-script`) so Christian Sogdian — Sogdian language in East Syriac script — trains and evaluates from a Syriac base model rather than the Sogdian default.

## Installation

```bash
git clone https://github.com/areopaguaworkshop/msocr.git
cd msocr
uv sync
```

The project targets Python 3.12 via `.python-version`.

## Workflow

```text
Manuscript image/PDF
        |
        +--> annotate/demo --> React UI + annotation API
        |                         |
        |                         +--> PAGE/ALTO XML or TSV ground truth
        |                                      |
        |                                      +--> frozen split manifest
        |                                               |
        |                         +---------------------+--------------------+
        |                         |                                          |
        |                   msocr train                              msocr train-remote
        |                 local ketos train                     enrich polygons + RunPod
        |                         |                                          |
        |                         +---------------------+--------------------+
        |                                               |
        |                                      Kraken .safetensors
        |                                               |
        |                                      msocr evaluate
        |                                      holdout CER/WER
        |
        +--> msocr htr / POST /htr
                 |
          NLbin + manuscript crop
                 |
          RTL baseline segmentation
                 |
          Kraken recognition
                 |
          JSON or Markdown
```

- **Ground truth:** `annotate`, `annotation-api`, and `demo` serve the same React SPA. Sessions can export PAGE XML, ALTO XML, or TSV.
- **Local training:** `train` compiles PAGE/ALTO XML to Arrow and invokes Kraken `ketos train`.
- **Remote training:** `train-remote` processes one manifest style group, enriches PAGE baselines with polygons, uploads train/validation data to RunPod, downloads the best `.safetensors` checkpoint, then evaluates the holdout locally.
- **Runtime:** `htr`, the API, and `runtime-smoke-check` share the same model resolver. The standalone `preprocess` command is optional and is not run automatically by HTR.

## Quick Start

### Run HTR on a manuscript image

```bash
uv run msocr htr --lang sogdian --model models/kraken/sogdian_manuscript.mlmodel /path/to/page.png
```

If you have a default local model, set one of these environment variables and omit `--model`:

```bash
export MSOCR_HTR_RUNTIME_MODEL_PATH=models/kraken/sogdian_manuscript.mlmodel
uv run msocr htr /path/to/page.png
```

### Write Markdown instead of JSON

```bash
uv run msocr htr /path/to/page.png --output-format markdown --output output/page.md
```

### Train a Kraken recognizer from PAGE/ALTO XML

```bash
uv run msocr train \
  --lang sogdian \
  --config msocr/configs/sogdian_config.yaml \
  --gt-dir data/sogdian/xml
```

You can also train from a frozen split manifest:

```bash
uv run msocr train \
  --split-manifest-id data/manifests/c2av-finetune.json \
  --split-partition train
```

### Preprocess manuscript images

```bash
uv run msocr preprocess --input-dir data/sogdian/images
```

Processed images are written to `data/sogdian/images/processed`.

### Handle fragmentary manuscripts

For damaged manuscripts with detached/rotated fragments, run the three-phase pipeline. Phase 1 isolates fragments (Sauvola + connected components + DBSCAN), Phase 2 binarizes each (Savola or NLbin with bleed-through detection), Phase 3 deskews each via Hough on the binary mask:

```bash
uv run msocr isolate-fragments path/to/page.tif --output-dir tmp/phase1_fragments/
uv run msocr binarize-fragments path/to/page.tif \
  --fragments-json tmp/phase1_fragments/fragments.json --output-dir tmp/phase2_binarized/
uv run msocr deskew-fragments path/to/page.tif \
  --fragments-json tmp/phase1_fragments/fragments.json \
  --binarized-dir tmp/phase2_binarized/ --output-dir tmp/phase3_deskewed/
```

To crop exact-count row bands from a page (with optional ROI and manual row centers):

```bash
uv run msocr extract-lines path/to/page.tif --expected-lines 18 --output-dir tmp/lines/
```

See `docs/kraken-fragmentary-manuscripts.md` for the full pipeline rationale.

### Run the API

```bash
export MSOCR_HTR_RUNTIME_MODEL_PATH=models/kraken/sogdian_manuscript.mlmodel
uv run msocr api --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /htr`

`POST /htr` accepts JSON with `image_path`, or multipart uploads with a `file`/`image` field plus optional `lang`, `variant`, `model`, and `device` fields.

### Validate runtime setup

```bash
uv run msocr runtime-smoke-check --lang sogdian
```

Run an end-to-end smoke check against a local image:

```bash
uv run msocr runtime-smoke-check \
  --lang sogdian \
  --image /path/to/page.png \
  --require-engine kraken
```

Probe a live API instance:

```bash
uv run msocr runtime-smoke-check \
  --base-url http://127.0.0.1:8000 \
  --image /path/to/page.png \
  --require-engine kraken
```

### Run the annotation UI (demo)

`msocr demo` now serves the annotation API + React SPA (the Gradio demo is gone):

```bash
uv run msocr demo --host 127.0.0.1 --port 8001
# → http://127.0.0.1:8001/
```

Requires the built SPA: `cd frontend && npm install && npm run build`.

### Run the annotation API

The annotation API stores Sogdian ground-truth sessions, page images, line crops, and annotations, and serves the React annotation SPA from `frontend/dist/`.

```bash
uv run msocr annotation-api --host 127.0.0.1 --port 8001 --base-dir msocr/data
```

If `frontend/dist/index.html` is missing, build the SPA first:

```bash
cd frontend && npm install && npm run build
```

Add `--no-crop-manuscript-area` to disable auto-detection and cropping of the manuscript area before line segmentation.

Exports supported by annotation sessions:

- ALTO XML
- PAGE XML
- TSV for Kraken training

### Annotate via the browser UI

`msocr annotate` starts the annotation API and prints the direct UI URL (React SPA; OpenSeadragon viewer, RTL line view, Sogdian Latin-transliteration palette):

```bash
uv run msocr annotate --host 127.0.0.1 --port 8001 --base-dir msocr/data
# → Annotation UI: http://127.0.0.1:8001/ui/{session_id}
```

It does not auto-open a browser (flaky in headless/Docker/SSH) — click the printed URL.

#### How to annotate a fragment (step by step)

The annotation editor produces **PAGE XML** — one annotation session that
trains both the Kraken recognizer (`ketos train`) and a future learned
segmenter (orli / YOLO-OBB / D-FINE). See
`docs/2026-08-10-fragment-segmentation-model-plan-v1.md` for the two-model
pipeline this annotation feeds.

**Step 0 — Preprocess the plate first (outside the browser).** The editor
expects an isolated, deskewed fragment image, not a raw publication plate:

```bash
uv run msocr isolate-fragments plate.png --output-dir tmp/p1/
uv run msocr binarize-fragments plate.png \
  --fragments-json tmp/p1/fragments.json --output-dir tmp/p2/
uv run msocr deskew-fragments plate.png \
  --fragments-json tmp/p1/fragments.json \
  --binarized-dir tmp/p2/ --output-dir tmp/p3/
# → deskewed fragment at tmp/p3/frag_001_deskewed.png
```

Optional line-count hint to guide your eye (classical CV, not a model):

```bash
uv run msocr extract-lines tmp/p3/frag_001_deskewed.png \
  --expected-lines 18 --output-dir tmp/lines/
# → line crops + overlay.jpg
```

**Step 1 — Create a session.** In the browser at
`http://127.0.0.1:8001`:

- **Language**: `sogdian`
- **Script variant**: `christian-syriac-script`
- **Fragment path**: absolute path to the deskewed fragment PNG (e.g.
  `/home/you/project/msocr/tmp/p3/frag_001_deskewed.png`)
- Submit → redirected to `/ui/{session_id}`

Fragmented-manuscript sessions start with **no baselines**. Default Kraken
BLLA failed the E27 pilot (zero usable line crops), so it does not
auto-suggest lines. Draw them manually.

**Step 2 — Draw regions (press `R`, optional but recommended).** Pick a
region type from the palette, click points around the area, double-click to
close. Region types:

| Label | When to use |
|---|---|
| Main | The primary text column |
| MainText | Marginal notes, glosses, commentary |
| Numbering | Folio / page / quire numbers |
| Damage | Physically damaged or unreadable area (Kraken will not read it) |
| Graphic | Illustrations, decorations, ornaments |
| DigitizationArtefact | Scan bleed-through, shadows, ruler marks |
| Custom | Anything that does not fit the above |

**Step 3 — Draw baselines (press `B`, the critical step).** A baseline is
the reading line under the text — Kraken reads upward from it.

1. Press `B`
2. Pick a line type (usually `Default`; `Heading` for rubrics;
   `Interlinear` for squeezed lines)
3. Click the **start** of the line — for RTL Sogdian this is the visual
   **right** end
4. Click the **end** of the line — visual left end
5. The baseline is created and selected

Two clicks per line. Repeat for every visible line fragment on the plate.

**Step 4 — Transcribe (press `T`).**

1. Press `T`
2. Click a baseline → the viewer centers on it, right panel activates
3. Type the Sogdian text, right-to-left
4. Use the character palette below the textarea (U+10F30–U+10F44) if you
   do not have a Sogdian keyboard
5. Press `Enter` to save + jump to next line (auto-saves on 2 s debounce)

The right panel shows all lines in reading order. Drag the `⠿` handle to
reorder. Reading order = top-to-bottom for Sogdian manuscript columns.

**Step 5 — Export PAGE XML.** Click **PAGE XML** in the top bar. The
browser downloads a `.page.xml` file containing one `<TextRegion>` per
region and one `<TextLine>` per baseline with `<Baseline>`, `<Coords>`,
and `<TextEquiv><Unicode>…</Unicode></TextEquiv>` carrying your
transcription. This is the file that trains both models.

#### Keyboard shortcuts

| Key | Action |
|---|---|
| `V` | Navigate mode (pan/zoom) |
| `R` | Region mode |
| `B` | Baseline mode |
| `T` | Transcribe mode |
| `Enter` (textarea) | Save + next line |
| `Ctrl+↑` / `Ctrl+↓` | Previous / next line |
| `Esc` | Cancel current drawing |
| `Del` / `Backspace` | Delete selected region or line |
| `Ctrl+S` | Save now |
| `?` (top bar) | Open the in-editor guide |

#### Critical rules for fragmented C2 manuscripts

1. **Do not join across holes.** Draw each contiguous visible piece as a
   separate baseline. A row interrupted by a lacuna = two baselines, linked
   by `row_id` metadata (see `docs/2026-08-02-christian-sogdian-fragment-htr-plan-v1.md` §1).
2. **Draw baselines under the text, not through it.** Kraken reads upward
   from the baseline.
3. **Transcribe what you see, not what you expect.** Damaged characters get
   a `` placeholder — do not guess.
4. **One session per page.** PAGE XML export is per-page.
5. **Reading order = line order in the right panel.** Drag to reorder
   before exporting.

See `docs/ANNOTATION.md` for the full in-editor guide with screenshots and
troubleshooting.

### Train a style-group on a RunPod GPU Cloud Pod

```bash
export RUNPOD_API_KEY=...
uv run msocr train-remote \
  --manifest data/manifests/c2av-finetune.json \
  --style-group c2av-syriac-finetune \
  --base-model models/kraken/sophro_mhiro_syriac.safetensors \
  --output-model models/kraken/c2av_finetune.safetensors \
  --reports-dir reports/ \
  --pod-gpu "NVIDIA GeForce RTX 3090" \
  --ssh-key ~/.ssh/id_ed25519 \
  --epochs 60 --min-epochs 8 --lag 10 \
  --freeze-backbone 999999 --warmup 200 --lr 1e-4 --augment
```

`--base-model` is optional when the style group has `base_model_override`; explicit CLI input wins. See `docs/runpod.md` for RunPod credentials and recovery notes.

### Evaluate a trained model

```bash
uv run msocr evaluate \
  --manifest data/manifests/c2av-finetune.json \
  --style-group c2av-syriac-finetune \
  --model models/kraken/c2av_finetune.safetensors \
  --reports-dir reports/
```

Writes `reports/{manifest_id}__{style_group}__{model_stem}.{json,md}` with per-manuscript and per-style-group CER/WER/Accuracy (parsed from `ketos test` stdout; no invented metrics).

## CLI Reference

### `htr`

```bash
uv run msocr htr [OPTIONS] INPUT_PATH
```

Key options:

- `--lang sogdian|old_sogdian` (default: `sogdian`)
- `--model PATH` optional Kraken `.mlmodel` or `.safetensors` override
- `--variant TEXT` metadata label, default `standard`
- `--output-format json|markdown`
- `--output PATH`
- `--device cpu|cuda|cuda:0|cuda:1`

### `train`

```bash
uv run msocr train [--config PATH] [--gt-dir DIR | --gt-file FILE | --split-manifest-id ID_OR_PATH]
```

Training uses `ketos compile` followed by `ketos train`. The default config is `msocr/configs/sogdian_config.yaml`.

### `train-remote`

```bash
uv run msocr train-remote --manifest PATH --style-group ID --base-model PATH --output-model PATH
```

Trains one style group on a RunPod GPU Cloud Pod, downloads the best `.safetensors` checkpoint, then evaluates locally. Reads `RUNPOD_API_KEY` from the environment. Defaults are smoke-oriented: RTX 3090, the official RunPod PyTorch image, 2 epochs, 0 minimum epochs, no augmentation, device `auto`, and 8 workers. Set production hyperparameters explicitly; use `--freeze-old-rows` only with a generated union-codec JSON.

### `evaluate`

```bash
uv run msocr evaluate --manifest PATH --style-group ID --model PATH --reports-dir DIR
```

Runs `ketos test` over a style-group's holdout partition, writes JSON + Markdown benchmark report. Thin wrapper — no invented metrics, reuses what `ketos test` reports.

### `evaluate-segmenter`

```bash
uv run msocr evaluate-segmenter --heldout-json PATH [--backend blla|kraken|yolo-obb] [--model PATH] [--recognizer PATH] [--reports-dir DIR] [--label TEXT]
```

Evaluates a line-level segmenter on a frozen held-out fragment set (Phase 2 of the fragment-segmentation plan). Produces `reports/{label}__segmenter.json` with baseline precision/recall/F1, split/merge counts, row grouping, gap localization, reading order, and optional end-to-end CER when `--recognizer` is provided.

`--heldout-json` is a JSON list of `{"id", "image", "gt_xml"}` entries you curate once and freeze — plates never used in any training split. `--backend blla` (default) loads the stock Kraken BLLA model as the zero-shot floor to beat; `--backend kraken --model path/to/seg.safetensors` loads any Kraken-plugin segmenter (orli / D-FINE / fine-tuned BLLA); `--backend yolo-obb --model best.pt` loads an Ultralytics YOLO-OBB model. See `docs/2026-08-10-fragment-segmentation-model-plan-v1.md` Phase 2.

### `demo`

```bash
uv run msocr demo [--host HOST] [--port PORT] [--share]
```

Launches the annotation API + React SPA (legacy Gradio demo removed; `--share` is a silent no-op). Requires `frontend/dist/index.html` — build it first with `cd frontend && npm install && npm run build`. Default port `8001`.

### `annotate`

```bash
uv run msocr annotate [--host HOST] [--port PORT] [--base-dir DIR] [--no-crop-manuscript-area]
```

Starts the annotation API (port 8001) and prints a direct `/ui/{session_id}` URL. Defaults: `127.0.0.1:8001`, base-dir `.`. Requires the built React SPA in `frontend/dist/`. Does not auto-open a browser (flaky in headless/Docker/SSH) — click the printed URL.

### `annotation-api`

```bash
uv run msocr annotation-api [--host HOST] [--port PORT] [--base-dir DIR] [--no-crop-manuscript-area]
```

Same annotation API as `annotate`, minus the URL hint. Default base-dir `msocr/data`.

### `extract-lines`

```bash
uv run msocr extract-lines IMAGE --expected-lines N --output-dir DIR [--roi L,T,R,B] [--row-centers Y,Y,...] [--min-component-area N]
```

Crops exactly N row-band line images from a page (with QA overlay + contact sheet).

### `isolate-fragments`

```bash
uv run msocr isolate-fragments IMAGE [--output-dir DIR] [--sauvola-window N] [--min-component-area N] [--min-fragment-area N] [--dbscan-eps N]
```

Phase 1 of the fragmentary-manuscript pipeline. Writes `fragments.json`, per-fragment PNGs, and an overlay JPG.

### `binarize-fragments`

```bash
uv run msocr binarize-fragments IMAGE --fragments-json PATH [--output-dir DIR] [--sauvola-window N]
```

Phase 2: per-fragment binarization (Sauvola or NLbin, auto-selected by bleed-through detection). Writes `{fragment_id}_mask.png` and `_compare.jpg`.

### `deskew-fragments`

```bash
uv run msocr deskew-fragments IMAGE --fragments-json PATH [--binarized-dir DIR] [--output-dir DIR]
```

Phase 3: per-fragment Hough deskew using Phase 2 masks. Writes deskewed crops plus `deskew_angles.json`.

### `runtime-smoke-check`

```bash
uv run msocr runtime-smoke-check [--image PATH] [--base-url URL]
```

Without `--base-url`, this validates local runtime model resolution and optionally runs Kraken on `--image`. With `--base-url`, it calls `/health` and `/htr` on a running API.

## Runtime Model Resolution

Model selection is local-only and intentionally simple:

1. explicit `--model` or API `model`
2. `MSOCR_HTR_RUNTIME_MODEL_PATH`
3. `MSOCR_HTR_MODEL_PATH`
4. `MSOCR_RUNTIME_MODEL_PATH`
5. `models/kraken/sogdian_manuscript.mlmodel`

The default path is a convention only; `models/` is gitignored, so place the model there yourself or use an explicit path.

## Ground Truth Manifests

Frozen manifests live under `data/manifests/` by convention and use manuscript-isolated partitions:

```json
{
  "manifest_id": "c2av-finetune",
  "language": "sogdian",
  "writing_mode": "handwritten",
  "script_block": "U+0710",
  "script_variant": "christian-syriac-script",
  "base_dir": "dataset/christian_sogdian_c2av",
  "partitions": {
    "train": [
      {"id": "c2av01-train", "manuscript_id": "c2av01", "xml_path": "gt/c2av01_page_01.xml"}
    ],
    "validation": [
      {"id": "c2av11-val", "manuscript_id": "c2av11", "xml_path": "gt/c2av11_page_11.xml"}
    ],
    "holdout": [
      {"id": "c2av12-holdout", "manuscript_id": "c2av12", "xml_path": "gt/c2av12_page_12.xml"}
    ]
  },
  "style_groups": {
    "c2av-syriac-finetune": {
      "manuscript_ids": ["c2av01", "c2av02", "c2av03", "c2av04", "c2av05", "c2av06", "c2av07", "c2av08", "c2av09", "c2av10", "c2av11", "c2av12"],
      "base_model_override": "models/kraken/sophro_mhiro_syriac.safetensors"
    }
  }
}
```

The loader rejects manifests where the same `manuscript_id` appears in more than one partition. `script_block` must be one of `U+10F30` (Sogdian) or `U+0710` (Syriac). `style_groups` maps a `style_group_id` to a list of `manuscript_ids` plus an optional `base_model_override` for per-style-group fine-tuning.

## Project Layout

```text
msocr/
├── cli.py
├── configs/sogdian_config.yaml
├── data/
│   ├── manifest.py
│   └── session_manager.py
├── datasets/splitter.py
├── evaluation/
│   ├── metrics.py
│   ├── harness.py
│   ├── layout_metrics.py      # baseline/row/gap metrics (evaluate-layout)
│   └── segmenter_harness.py   # model-neutral segmenter eval (evaluate-segmenter)
├── language_registry.py
├── models/inference.py
├── output/formats.py
├── preprocessing/
│   ├── preprocessor.py
│   ├── binarize.py        # Sauvola / NLbin with bleed-through detection
│   └── deskew.py          # per-fragment Hough deskew
├── segmentation/
│   ├── fragment_isolation.py  # Sauvola + CC + DBSCAN
│   └── row_bands.py           # exact-count line-band extraction
├── service/
│   ├── api.py
│   ├── annotation_api.py     # serves React SPA from frontend/dist/
│   ├── deploy.py
│   └── runtime.py
├── training/
│   ├── ketos_trainer.py
│   ├── orchestrator.py       # polygon enrichment + per-style-group walker
│   └── runpod_runner.py      # RunPod SSH host:port discovery
└── utils/
frontend/                       # React + OpenSeadragon annotation SPA
├── src/{AnnotateEditor,PlateGallery,SessionList,App,types}.tsx
└── dist/                      # build output, served by annotation-api
data/manifests/
├── c2av-finetune.json                    # active 10/1/1 train/validation/holdout split
├── c2av-{smoke,eval-train,eval-val}.json
├── christian-sogdian-c2av*.json
├── e57d259b-smoke.json
├── ms-jer-36-adapt.json
└── lopo/                                 # 12 leave-one-plate-out manifests
docs/
├── runpod.md                              # RunPod credentials and recovery
├── kraken-fragmentary-manuscripts.md      # fragment pipeline rationale
├── kraken-training-data-research.md
├── multi-script-htr-research.md
├── runpod-gpu-research.md
├── codeql-scan.md  semgrep-scan.md
└── plans/
```

Remote training artifacts:

- `Dockerfile.train` — optional custom RunPod image
- `docs/runpod.md` — credentials and manual recovery notes
- `data/manifests/c2av-finetune.json` — active Christian Sogdian fine-tune split
- `data/manifests/lopo/` — leave-one-plate-out split variants; run one manifest/style group at a time

## Tests

```bash
uv run pytest
```

Targeted examples:

```bash
uv run pytest tests/service/test_runtime.py tests/service/test_deploy.py
uv run pytest tests/data/test_manifest.py tests/data/test_session_manager.py
```

The only GitHub Actions workflow currently runs PR Agent review automation; it does not run the test suite. Run tests locally before merging.

## Notes

- All runtime recognition is Kraken HTR.
- Sogdian is the only supported language in the registry and CLI choices. Two script blocks are tracked for training: Sogdian `U+10F30` and Syriac `U+0710` (Christian Sogdian uses the latter).
- Remote training uses RunPod GPU Cloud Pods via `msocr train-remote`; evaluation wraps `ketos test` (no invented metrics).
- JSON and Markdown are the only HTR output formats.
- PDF input is rendered to images before HTR; searchable PDF output is intentionally not part of this focused runtime.
- The annotation UI is a React SPA under `frontend/`; build it (`cd frontend && npm install && npm run build`) before running `annotate`, `annotation-api`, or `demo`.
- `models/`, generated `output/` artifacts, and `msocr/data/sessions/` (local annotation sessions) are gitignored.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
