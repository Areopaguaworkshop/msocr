# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

`msocr` is a focused toolkit for handwritten Sogdian manuscript recognition. It uses local [Kraken](https://github.com/mittagessen/kraken) models for HTR and includes the supporting tools needed to prepare damaged pages, annotate RTL text, train locally or on RunPod, evaluate fixed holdouts, and serve recognition through a CLI or FastAPI.

The runtime is intentionally narrow:

- Sogdian only (`sogdian`, with `old_sogdian` as an alias)
- handwritten text only
- local Kraken inference only
- JSON or Markdown output only
- no Tesseract, OCRmyPDF, printed-OCR routing, searchable PDF, or model registry

Christian Sogdian is represented as Sogdian language in the Syriac Unicode block (`U+0710`); manuscripts in the Sogdian block use `U+10F30`. Both are read right-to-left.

## Project status

The current C2AV experiment is a working research baseline, not a production-quality recognizer. The best recorded 200-epoch fine-tune reached 71.43% holdout CER, while train CER reached 6.08%. That gap shows severe overfitting: more reviewed pages from more manuscripts are the next useful improvement, not more infrastructure.

Experiment reports live in [`reports/`](reports/) and detailed findings live in [`docs/`](docs/). In particular:

- [`docs/fixa-v3-cer-lower.md`](docs/fixa-v3-cer-lower.md) — current recognition experiment
- [`docs/ANNOTATION.md`](docs/ANNOTATION.md) — annotation guide
- [`docs/kraken-fragmentary-manuscripts.md`](docs/kraken-fragmentary-manuscripts.md) — fragment preparation
- [`docs/runpod.md`](docs/runpod.md) — remote training setup and recovery

## Installation

The repository targets Python 3.12 via `.python-version`; package metadata permits Python 3.10 through 3.13.

```bash
git clone https://github.com/areopaguaworkshop/msocr.git
cd msocr
uv sync
```

Build the annotation SPA before using `annotate`, `annotation-api`, or `demo`:

```bash
cd frontend
npm install
npm run build
cd ..
```

Kraken models are not included in Git. Put compatible `.mlmodel` or `.safetensors` files under `models/kraken/`, or pass their path explicitly.

## Workflow

There are two model-training loops—recognition and layout—and one runtime recognition path.

```text
GROUND TRUTH AND TRAINING

Manuscript source (upload, local file, IIIF, or DTA inventory)
        |
        v
Optional fragment preparation
isolate -> deskew -> geometry mask -> review-only line proposals
        |
        v
React annotation UI + FastAPI session store
regions + RTL baselines + reading order + transcripts
        |
        v
Training PAGE XML + export audit
        |
        v
Frozen manifest: train / validation / holdout by manuscript
        |
        +---------------------------+
        |                           |
        v                           v
Local `msocr train`          `msocr train-remote`
ketos compile + train        PAGE filtering + polygon enrichment
        |                    -> RunPod ketos train -> download best model
        +---------------------------+
                                    |
                                    v
                            `msocr evaluate`
                            holdout CER / WER reports
                                    |
                                    v
                           reviewed Kraken model

LAYOUT-CANDIDATE TRAINING (SEPARATE MODEL)

Frozen manuscript-level train / validation manifest
        |
        v
`msocr train-segmenter-remote --backend ...`
safe PAGE staging + missing line-box synthesis + provenance hashes
        |
        +--> blla: ketos segtrain
        +--> orli: PAGE -> Arrow -> orli train
        +--> dfine: PAGE -> dfine train
        +--> yolo-obb: PAGE -> normalized OBB labels -> yolo obb train
        |
        v
RunPod GPU -> download best .safetensors/.pt
        |
        v
`msocr evaluate-segmenter` on the untouched layout holdout

RUNTIME RECOGNITION

Image or PDF / POST /htr
        |
        v
PDF page rendering (when needed)
        |
        v
Kraken NLbin -> manuscript-area crop -> RTL baseline segmentation
        |
        v
Kraken line recognition
        |
        v
JSON or Markdown
```

### Layout and line-segmentation status

Layout segmentation is a separate model stage from text recognition. The repository can now prepare and train all four planned candidates on RunPod, but no fragment-specific model has yet been trained, selected, or integrated into runtime HTR. Runtime recognition therefore still uses Kraken's stock BLLA automatically.

| Path | What runs now | Status |
|---|---|---|
| `msocr htr` and `POST /htr` | Stock Kraken BLLA, then the selected Kraken recognizer | Implemented; no custom segmenter option |
| Annotation `autosuggest` | Stock Kraken BLLA | Implemented; suggestions require human correction |
| `prepare-fragment-page --propose-lines` | Stock BLLA, or an explicitly supplied Kraken-compatible `--segmentation-model` | Implemented; output is review-only and not training-eligible |
| `extract-lines` | Classical connected-component row bands with a human-supplied line count | Implemented deterministic fallback; not a learned model |
| `evaluate-layout` | Compares an existing prediction JSON with reviewed PAGE XML | Implemented evaluator; it does not run a model |
| `evaluate-segmenter` | Runs stock BLLA, a supplied Kraken plugin model, or a supplied YOLO-OBB model on a frozen held-out set | Implemented evaluation harness |
| `train-segmenter-remote --backend blla` | PAGE XML → Kraken `ketos segtrain` | Implemented RunPod trainer; supports scratch or `--base-model` fine-tuning |
| `train-segmenter-remote --backend orli` | PAGE XML → Orli Arrow → `orli train` | Implemented RunPod fine-tuner; requires Orli base weights |
| `train-segmenter-remote --backend dfine` | PAGE XML → `dfine train` with line-only class mapping | Implemented RunPod trainer; supports scratch or base weights |
| `train-segmenter-remote --backend yolo-obb` | PAGE line polygons/baselines → normalized OBB labels → `yolo obb train` | Implemented RunPod fine-tuner; requires a YOLO `.pt` base model |
| Trained candidate and runtime selection | No trained model or recorded segmenter report is present | Not done; train, evaluate, then integrate only the winner |

The planned selection process is: annotate enough fragment-aware PAGE XML, freeze a held-out layout set, record stock BLLA and `row_bands` baselines, then train and compare fragment-specific candidates using baseline F1, split/merge errors, reading order, and end-to-end CER. A candidate is integrated only if it improves both geometry and recognition. Damage/ink segmentation remains a separate proposed experiment and is not in the runtime path.

### 1. Acquire or prepare source pages

The tracked E27 inventory can be downloaded or checksum-verified without committing the copyrighted images:

```bash
uv run msocr download-e27-images
uv run msocr download-e27-images --verify-only
```

For one mounted archive photograph, the integrated pipeline isolates the manuscript, deskews it, produces geometry masks and QA artifacts, and optionally runs review-only BLLA/Orli line proposals:

```bash
uv run msocr prepare-fragment-page page.jpg \
  --output-dir dataset/christian_sogdian_c2av/derived

uv run msocr prepare-fragment-page page.jpg \
  --output-dir dataset/christian_sogdian_c2av/derived \
  --propose-lines
```

Use `--isolation-mode components` when one page contains multiple detached fragments. The older manual stages remain available when intermediate control is needed:

```bash
uv run msocr isolate-fragments page.tif --output-dir tmp/fragments/
uv run msocr binarize-fragments page.tif \
  --fragments-json tmp/fragments/fragments.json \
  --output-dir tmp/binarized/
uv run msocr deskew-fragments page.tif \
  --fragments-json tmp/fragments/fragments.json \
  --binarized-dir tmp/binarized/ \
  --output-dir tmp/deskewed/
```

If the line count is known, classical row-band extraction can produce exact-count crops and a QA overlay:

```bash
uv run msocr extract-lines fragment.png \
  --expected-lines 18 \
  --output-dir tmp/lines/
```

These preparation commands are annotation aids. `msocr htr` does not automatically run this fragment pipeline.

### 2. Annotate ground truth

Start the annotation service and open the printed URL:

```bash
uv run msocr annotate \
  --host 127.0.0.1 \
  --port 8001 \
  --base-dir msocr/data
```

The React/OpenSeadragon UI supports browser uploads, local paths, and IIIF sources. It stores each session under `<base-dir>/sessions/<session-id>/` with the page image, session metadata, line crops, regions, baselines, gaps, reading order, and transcripts.

The annotation API can export PAGE XML, ALTO XML, or TSV. Kraken training uses the PAGE training export: it filters ineligible lines and writes a JSON audit beside the XML. Do not join a baseline across a lacuna; annotate each visible piece and link row/gap metadata in the editor.

Other launch forms use the same application:

```bash
uv run msocr annotation-api --host 127.0.0.1 --port 8001 --base-dir msocr/data
uv run msocr demo --host 127.0.0.1 --port 8001
```

`demo` is a legacy command name for the React annotation app; Gradio is no longer the active UI.

### 3. Freeze the data split

Training manifests live under `data/manifests/`. They keep entire manuscripts in separate train, validation, and holdout partitions and optionally group related hands under a `style_group_id`:

```json
{
  "manifest_id": "example-finetune",
  "writing_mode": "handwritten",
  "language": "sogdian",
  "script_block": "U+0710",
  "script_variant": "christian-syriac-script",
  "base_dir": "dataset/example",
  "partitions": {
    "train": [{"id": "p1", "manuscript_id": "ms-1", "xml_path": "gt/p1.xml"}],
    "validation": [{"id": "p2", "manuscript_id": "ms-2", "xml_path": "gt/p2.xml"}],
    "holdout": [{"id": "p3", "manuscript_id": "ms-3", "xml_path": "gt/p3.xml"}]
  },
  "style_groups": {
    "example-hand": {
      "manuscript_ids": ["ms-1", "ms-2", "ms-3"],
      "base_model_override": "models/kraken/base.safetensors"
    }
  }
}
```

The loader rejects a `manuscript_id` that leaks across partitions. Valid script blocks are `U+10F30` and `U+0710`.

### 4. Train

Local training compiles PAGE/ALTO XML to Kraken Arrow data and runs `ketos train` using `msocr/configs/sogdian_config.yaml` by default:

```bash
uv run msocr train --gt-dir data/sogdian/xml

uv run msocr train \
  --split-manifest-id data/manifests/c2av-finetune.json \
  --split-partition train
```

Remote training handles one style group at a time. It creates training-safe PAGE XML, enriches baselines with polygons required by Kraken 7, uploads XML/image pairs, runs `ketos train` on a RunPod GPU Cloud Pod, downloads the best checkpoint, and evaluates the holdout locally:

```bash
export RUNPOD_API_KEY=...

uv run msocr train-remote \
  --manifest data/manifests/c2av-finetune.json \
  --style-group c2av-syriac-finetune \
  --output-model models/kraken/c2av_finetune.safetensors \
  --reports-dir reports/ \
  --ssh-key ~/.ssh/id_ed25519 \
  --epochs 60 --min-epochs 8 --lag 10 \
  --freeze-backbone 999999 --warmup 200 --lr 1e-4 --augment
```

An explicit `--base-model` overrides the style group's `base_model_override`. Without either, the script-block default is used when available; otherwise training starts from scratch. The command defaults are smoke-test oriented, so set production hyperparameters explicitly.

Train a layout candidate with the same frozen train/validation split. Start with `--dry-run`: it performs all validation and conversion, rejects identical XML/image content across train and validation, and writes the exact pinned setup/training commands without reading `RUNPOD_API_KEY` or creating a paid pod.

```bash
uv run msocr train-segmenter-remote \
  --manifest data/manifests/layout-fragments.json \
  --backend blla \
  --output-model models/kraken/fragment_blla.safetensors \
  --reports-dir reports/ \
  --dry-run

export RUNPOD_API_KEY=...
chmod 600 ~/.ssh/id_ed25519

uv run msocr train-segmenter-remote \
  --manifest data/manifests/layout-fragments.json \
  --backend yolo-obb \
  --base-model /path/to/yolo11n-obb.pt \
  --output-model models/layout/fragment_yolo11_obb.pt \
  --epochs 100 --image-size 1280 --augment
```

Use `--backend blla`, `orli`, `dfine`, or `yolo-obb`. Orli and YOLO-OBB require `--base-model`; BLLA and D-FINE may start from scratch. If a PAGE `TextLine` has no `Coords`, `--line-height` supplies the only calibrated assumption used to turn its baseline into a training box. Inputs are parsed with DTD/entity/network access disabled, bundled under generated names, and accompanied by SHA-256 provenance in `reports/{job}-training.json`. The command pins Kraken 7.0.2, Orli 0.0.2, `dfine_kraken` 0.4.3, or Ultralytics 8.4.82 on the pod.

### 5. Evaluate and iterate

Evaluate a recognizer against the style group's fixed holdout:

```bash
uv run msocr evaluate \
  --manifest data/manifests/c2av-finetune.json \
  --style-group c2av-syriac-finetune \
  --model models/kraken/c2av_finetune.safetensors \
  --reports-dir reports/
```

This writes `reports/{manifest_id}__{style_group}__{model_stem}.{json,md}` with the metrics reported by `ketos test`.

Layout proposals can be checked separately:

```bash
uv run msocr evaluate-layout \
  --ground-truth reviewed.page.xml \
  --predictions candidate.segments.json \
  --report reports/layout.json

uv run msocr evaluate-segmenter \
  --heldout-json data/heldout-fragments.json \
  --backend blla \
  --reports-dir reports/
```

`evaluate-segmenter` reports baseline precision/recall/F1, split/merge counts, row grouping, gap localization, and reading order. Add `--recognizer MODEL` to include end-to-end CER.

### 6. Run HTR

Model resolution uses the first available source:

1. explicit CLI/API model path
2. `MSOCR_HTR_RUNTIME_MODEL_PATH`
3. `MSOCR_HTR_MODEL_PATH`
4. `MSOCR_RUNTIME_MODEL_PATH`
5. `models/kraken/sogdian_manuscript.mlmodel`

Run an image or PDF through the CLI:

```bash
uv run msocr htr \
  --model models/kraken/sogdian_manuscript.mlmodel \
  --output-format json \
  --output output/page.json \
  /path/to/page.png
```

PDF input is rendered to temporary page images before recognition. Use `--output-format markdown` for Markdown output.

Validate model resolution or run an end-to-end local smoke check:

```bash
uv run msocr runtime-smoke-check --model models/kraken/sogdian_manuscript.mlmodel

uv run msocr runtime-smoke-check \
  --model models/kraken/sogdian_manuscript.mlmodel \
  --image /path/to/page.png \
  --require-engine kraken
```

### 7. Serve the runtime API

```bash
export MSOCR_HTR_RUNTIME_MODEL_PATH=models/kraken/sogdian_manuscript.mlmodel
uv run msocr api --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /htr`

`POST /htr` accepts either JSON with a server-local `image_path` or a multipart upload:

```bash
curl -X POST http://127.0.0.1:8000/htr \
  -H 'content-type: application/json' \
  -d '{"image_path":"/absolute/path/page.png","lang":"sogdian"}'

curl -X POST http://127.0.0.1:8000/htr \
  -F file=@/path/to/page.png \
  -F lang=sogdian
```

Probe a running service with the same smoke-check command:

```bash
uv run msocr runtime-smoke-check \
  --base-url http://127.0.0.1:8000 \
  --image /path/to/page.png \
  --require-engine kraken
```

## CLI summary

Run `uv run msocr COMMAND --help` for complete options.

| Command | Purpose |
|---|---|
| `download-e27-images` | Download or verify the tracked DTA E27 image inventory |
| `prepare-fragment-page` | Integrated fragment isolation, deskew, masking, and optional line proposals |
| `isolate-fragments` | Manual fragment pipeline: isolate fragments |
| `binarize-fragments` | Manual fragment pipeline: create per-fragment masks |
| `deskew-fragments` | Manual fragment pipeline: deskew fragment crops |
| `extract-lines` | Produce exact-count row-band crops and QA images |
| `preprocess` | Apply the standalone directory preprocessing pipeline |
| `annotate` | Start the annotation app and print its URL |
| `annotation-api` | Start the annotation API and React SPA |
| `demo` | Legacy alias-like launcher for the annotation app |
| `train` | Compile XML and train locally with Kraken ketos |
| `train-remote` | Train one manifest style group on RunPod, download, and evaluate |
| `train-segmenter-remote` | Train one of four layout candidates on RunPod; supports safe dry-run planning |
| `evaluate` | Evaluate recognition on a manifest holdout |
| `evaluate-layout` | Compare saved layout predictions with reviewed PAGE geometry |
| `evaluate-segmenter` | Evaluate a segmenter on a frozen held-out set |
| `dump-preds` | Bootstrap annotation by predicting transcripts for PAGE lines |
| `htr` | Recognize an image or PDF locally |
| `api` | Serve `GET /health` and `POST /htr` |
| `runtime-smoke-check` | Validate a local model path or a live API |

## Project layout

```text
msocr/
├── cli.py                    # Click command surface
├── language_registry.py      # Sogdian profile, aliases, script defaults
├── models/inference.py       # NLbin, RTL segmentation, Kraken recognition
├── preprocessing/            # Page/fragment image preparation
├── segmentation/             # Fragment, row-band, and BLLA helpers
├── data/
│   ├── manifest.py           # Frozen split loading and validation
│   └── session_manager.py    # Annotation persistence and exports
├── training/
│   ├── ketos_trainer.py      # Local ketos wrapper
│   ├── orchestrator.py       # One-style-group remote workflow
│   ├── segmenter_trainer.py  # Four layout backends, conversion, provenance
│   ├── page_export.py        # Training PAGE filtering and audit
│   └── runpod_runner.py      # Pod lifecycle, SSH, upload/download
├── evaluation/               # Recognition and layout metrics
└── service/
    ├── api.py                # Runtime FastAPI service
    ├── annotation_api.py     # Annotation API and SPA hosting
    └── runtime.py            # Shared CLI/API model resolution

frontend/                     # React + OpenSeadragon annotation SPA
data/manifests/               # Frozen train/validation/holdout definitions
dataset/                      # Local manuscript data
models/                       # Gitignored Kraken models
reports/                      # Evaluation and experiment reports
docs/                         # Research, plans, and operational guides
tests/                        # Pytest suite
```

## Development

```bash
uv run pytest
cd frontend && npm run build
```

For a targeted backend check:

```bash
uv run pytest tests/service/test_runtime.py tests/data/test_manifest.py
```

The GitHub Actions configuration currently runs PR Agent automation, not the test suite. Run checks locally before merging.

## License

Licensed under the [Apache License 2.0](LICENSE).
