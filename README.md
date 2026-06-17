# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

`msocr` is now a focused Sogdian manuscript HTR toolkit. It uses Kraken for local handwritten text recognition, keeps language handling Sogdian-only, and provides small tools for ground-truth preparation, model training (local or remote on RunPod GPU Cloud Pods), inference, an API, a browser-based annotation UI, and a Gradio demo.

Active scope (remote training): RunPod GPU Cloud Pod submission for `ketos train` fine-tuning via `msocr train-remote`, with a minimal procedural per-style-group orchestrator (one style-group at a time, not a DAG engine).

Removed scope: printed OCR routing, Tesseract/OCRmyPDF fallbacks, benchmark promotion flows, artifact registry publication, and multi-language orchestration.

## Supported Language

| Code | Alias | Direction | Runtime font | Default model |
|---|---|---|---|---|
| `sogdian` | `old_sogdian` | RTL | Noto Sans Sogdian | `models/kraken/sogdian_manuscript.mlmodel` |

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

### Run the Gradio demo

```bash
export MSOCR_HTR_RUNTIME_MODEL_PATH=models/kraken/sogdian_manuscript.mlmodel
uv run msocr demo --host 127.0.0.1 --port 7860
```

### Run the annotation API

The annotation API stores Sogdian ground-truth sessions, page images, line crops, and annotations.

```bash
uv run msocr annotation-api --host 127.0.0.1 --port 8001 --base-dir msocr/data
```

Exports supported by annotation sessions:

- ALTO XML
- PAGE XML
- TSV for Kraken training

### Annotate via the browser UI

`msocr annotate` starts the annotation API and prints the `/ui` URL (HTMX + Alpine.js, no build step, vendored JS):

```bash
uv run msocr annotate --host 127.0.0.1 --port 8001 --base-dir msocr/data
# → Annotation UI: http://127.0.0.1:8001/ui
```

Open `/plan` for the design doc view; `/ui/{session_id}/{line_n}` for the RTL line annotation view with the Sogdian Unicode palette.

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

### `annotate`

```bash
uv run msocr annotate [--host HOST] [--port PORT] [--base-dir DIR]
```

Starts the annotation API (port 8001) and prints the `/ui` URL. Defaults: `127.0.0.1:8001`, base-dir `.`. Does not auto-open a browser (flaky in headless/Docker/SSH) — click the printed URL.

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
├── preprocessing/preprocessor.py
├── segmentation/
├── service/
│   ├── api.py
│   ├── annotation_api.py
│   ├── annotation_ui/        # vendored htmx+alpine + Jinja2 templates
│   ├── deploy.py
│   ├── gradio_demo.py
│   └── runtime.py
├── training/
│   ├── ketos_trainer.py
│   ├── orchestrator.py       # procedural per-style-group walker
│   └── runpod_runner.py      # RunPod GPU Cloud Pod runner
└── utils/
```

Remote training artifacts:

- `Dockerfile.train` — RunPod pod image (python:3.12-slim + uv + kraken 7.0)
- `docs/runpod.md` — runbook for `msocr train-remote`
- `data/manifests/berlin-turfan-{sogdian,syriac}-v1.json` — schema-only manifests (contents filled during data collection)

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
- Sogdian is the only supported language in the registry and CLI choices. Two script blocks are tracked for training: Sogdian `U+10F30` and Syriac `U+0710`.
- Remote training uses RunPod GPU Cloud Pods via `msocr train-remote`; evaluation wraps `ketos test` (no invented metrics).
- JSON and Markdown are the only HTR output formats.
- PDF input is rendered to images before HTR; searchable PDF output is intentionally not part of this focused runtime.
- `models/` and generated `output/` artifacts are gitignored.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
