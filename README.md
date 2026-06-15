# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

`msocr` is now a focused Sogdian manuscript HTR toolkit. It uses Kraken for local handwritten text recognition, keeps language handling Sogdian-only, and provides small tools for ground-truth preparation, model training, inference, an API, and a Gradio demo.

Removed scope: printed OCR routing, Tesseract fallbacks, benchmark promotion flows, remote training submission, artifact registry publication, and multi-language orchestration.

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
  "manifest_id": "sogdian-htr-v1",
  "language": "sogdian",
  "writing_mode": "handwritten",
  "base_dir": "data/sogdian",
  "partitions": {
    "train": [
      {"id": "line_0001", "xml_path": "ms001/line_0001.xml", "manuscript_id": "ms001"}
    ],
    "validation": [],
    "holdout": []
  }
}
```

The loader rejects manifests where the same `manuscript_id` appears in more than one partition.

## Project Layout

```text
msocr/
├── cli.py
├── configs/sogdian_config.yaml
├── data/
│   ├── manifest.py
│   └── session_manager.py
├── datasets/splitter.py
├── evaluation/metrics.py
├── language_registry.py
├── models/inference.py
├── output/formats.py
├── preprocessing/pipeline.py
├── segmentation/
├── service/
│   ├── api.py
│   ├── annotation_api.py
│   ├── deploy.py
│   ├── gradio_demo.py
│   └── runtime.py
├── training/
└── utils/
```

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
- Sogdian is the only supported language in the registry and CLI choices.
- JSON and Markdown are the only HTR output formats.
- PDF input is rendered to images before HTR; searchable PDF output is intentionally not part of this focused runtime.
- `models/` and generated `output/` artifacts are gitignored.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
