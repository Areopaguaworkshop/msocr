# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

`msocr` is now a focused Sogdian manuscript HTR toolkit. It uses Kraken for local handwritten text recognition, keeps language handling Sogdian-only, and provides small tools for ground-truth preparation, model training (local or remote on RunPod GPU Cloud Pods), inference, an API, a browser-based annotation UI, and a Gradio demo.

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

- **Christian Sogdian manifests** under `data/manifests/`: `christian-sogdian-c2av.json` (canonical, growing plate set) plus two single-page smoke manifests (`e57d259b-smoke.json`, `christian-sogdian-c2av-image19-smoke.json`). Style group `christian-syriac-script-c2av` targets fine-tuning from a Syriac base model (`models/kraken/sophro_mhiro_syriac.mlmodel`).
- **Fragment-handling CLI pipeline** for damaged/fragmentary manuscripts: `msocr isolate-fragments` (Sauvola + connected components + DBSCAN), `binarize-fragments` (per-fragment Sauvola/NLbin with bleed-through detection), `deskew-fragments` (Hough deskew on binary masks), and `extract-lines` (exact-count row-band cropping with optional ROI / row-centers). See `docs/kraken-fragmentary-manuscripts.md`.
- **Annotation UI rewrite**: the browser UI moved from HTMX/Alpine to a React SPA (OpenSeadragon viewer + Phosphor icons, Sogdian Latin-transliteration palette). The SPA bundle lives in `frontend/dist/` and is served by the annotation API. Build it with `cd frontend && npm install && npm run build`. `msocr annotate`, `annotation-api`, and `demo` all refuse to start if `frontend/dist/index.html` is missing.
- **`msocr demo`** now launches the annotation API + React SPA on port 8001 (default). The legacy Gradio demo is dead code; `--share` is a silent no-op.
- **`--no-crop-manuscript-area`** flag on `annotate`, `annotation-api`, and `demo` disables automatic manuscript-area cropping before line segmentation.
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
  --split-manifest-id data/manifests/sogdian-htr-v1.json \
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

### Train a style-group on a RunPod GPU Cloud Pod

```bash
export RUNPOD_API_KEY=...
uv run msocr train-remote \
  --manifest data/manifests/berlin-turfan-sogdian-v1.json \
  --style-group manichaean-early \
  --base-model models/kraken/openiti-arabic-base.safetensors \
  --output-model models/kraken/sogdian-manichaean-early.mlmodel \
  --reports-dir reports/ \
  --pod-gpu "RTX 4090" \
  --pod-image msocr-kraken7:latest \
  --ssh-key ~/.ssh/id_ed25519 \
  --epochs 50 --min-epochs 20 --lag 10 --freeze-backbone 5000 --augment
```

See `docs/runpod.md` for the full runbook (API key, SSH key, pod image build/push, manual recovery, cost).

### Evaluate a trained model

```bash
uv run msocr evaluate \
  --manifest data/manifests/berlin-turfan-sogdian-v1.json \
  --style-group manichaean-early \
  --model models/kraken/sogdian-manichaean-early.mlmodel \
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
- `--model PATH` optional Kraken `.mlmodel` override
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

Trains one style-group on a RunPod GPU Cloud Pod, then evaluates locally. Reads `RUNPOD_API_KEY` from env. Key options: `--pod-gpu` (default `RTX 4090`), `--pod-image` (default `msocr-kraken7:latest`), `--ssh-key` (default `~/.ssh/id_ed25519`), `--epochs` (50), `--min-epochs` (20), `--lag` (10), `--freeze-backbone` (5000), `--augment/--no-augment`, `--device` (`cuda:0`), `--workers` (8). See `docs/runpod.md`.

### `evaluate`

```bash
uv run msocr evaluate --manifest PATH --style-group ID --model PATH --reports-dir DIR
```

Runs `ketos test` over a style-group's holdout partition, writes JSON + Markdown benchmark report. Thin wrapper — no invented metrics, reuses what `ketos test` reports.

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
  "manifest_id": "berlin-turfan-sogdian-v1",
  "language": "sogdian",
  "writing_mode": "handwritten",
  "script_block": "U+10F30",
  "base_dir": "data/berlin_turfan/sogdian",
  "partitions": {
    "train": [
      {"id": "line_0001", "xml_path": "ms001/line_0001.xml", "manuscript_id": "ms001", "image": "ms001/line_0001.tif"}
    ],
    "validation": [],
    "holdout": []
  },
  "style_groups": {
    "manichaean-early": {"manuscript_ids": ["ms001", "ms002"], "base_model_override": "openiti-arabic-base"}
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
│   └── harness.py
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
├── berlin-turfan-sogdian-v1.json         # schema-only
├── christian-sogdian-c2av.json           # canonical Christian Sogdian manifest
├── christian-sogdian-c2av-image19-smoke.json
└── e57d259b-smoke.json
docs/
├── runpod.md                              # RunPod runbook
├── kraken-fragmentary-manuscripts.md      # fragment pipeline rationale
├── kraken-training-data-research.md
├── multi-script-htr-research.md
├── runpod-gpu-research.md
├── codeql-scan.md  semgrep-scan.md
└── plans/
```

Remote training artifacts:

- `Dockerfile.train` — RunPod pod image (python:3.12-slim + uv + kraken 7.0)
- `docs/runpod.md` — runbook for `msocr train-remote`
- `data/manifests/berlin-turfan-{sogdian,syriac}-v1.json` — schema-only manifests (contents filled during data collection)
- `data/manifests/christian-sogdian-c2av*.json` — Christian Sogdian (East Syriac script) manifests, real and smoke variants

## Tests

```bash
uv run pytest
```

Targeted examples:

```bash
uv run pytest tests/service/test_runtime.py tests/service/test_deploy.py
uv run pytest tests/data/test_manifest.py tests/data/test_session_manager.py
```

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
