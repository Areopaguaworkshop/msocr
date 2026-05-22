# msocr

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

Manuscript OCR/HTR toolkit with route-aware language handling, Kraken/Tesseract integration, benchmark reporting, FastAPI service APIs, and Gradio demo UI.

## Current Status

### Implemented
- CLI with subcommands:
  - `ocr` (printed OCR, image/PDF input)
  - `htr` (handwritten OCR/HTR, image/PDF input)
  - `train` (Kraken ketos training)
  - `preprocess`
  - `benchmark`
  - `runpod-submit` (manifest-aware RunPod pod submission, dry-run by default)
  - `har-publish` (Harness Artifact Registry model bundle publication, dry-run by default)
  - `pipeline-submit` (end-to-end manifest-aware submit, benchmark, and promotion workflow)
  - `runtime-smoke-check` (printed or handwritten runtime validation for local resolution or live HTTP routes)
  - `api`
  - `demo`
- Printed OCR routing:
   - Greek: Kraken primary + Kraken fallback models
   - Latin: Kraken CATMuS-Print Large (2024-01-30, 98.56% accuracy, CER 1.44%) primary + Tesseract fallback
   - Syriac: Tesseract `syr` baseline with optional Serto/East CER-gated traineddata switch
   - Coptic: Tesseract (`cop`) route
   - Armenian: Tesseract (`hye-calfa-n` preferred, fallback `hye`)
   - Geez: Tesseract fallback chain (`gez` -> `tir` -> `amh`)
- Handwritten defaults:
   - Latin: Kraken CATMuS Medieval (8-15th century manuscripts)
   - Greek: Kraken greek-german serifs model
   - Syriac: HAR-backed Kraken runtime when a promoted handwritten `.mlmodel` is configured; otherwise fallback Transkribus bridge
- Printed benchmark runner with CER/WER JSON reports
- FastAPI backend (`msocr/service/api.py`)
- Gradio browser demo (`msocr/service/gradio_demo.py`)
- HAR-backed printed runtime model resolution for API and Gradio startup
- HAR-backed handwritten runtime model resolution for API, CLI, and Gradio using `{lang}-{variant}-handwritten` package naming
- Live runtime smoke helpers for `/ocr`, `/htr`, and Gradio root readiness
- Delegate examples for printed runtime and Syriac handwritten runtime deployment under `pipeline/harness/`

### Planned
- Handwritten benchmark/evaluation parity inside `pipeline-submit` so HTR artifacts can be gated and promoted end-to-end without dropping to a dedicated Harness YAML.
- Production handwritten workflows for Coptic/Armenian/Geez with local Kraken training from PAGE/ALTO exports.
- Expanded language-aware classifier/router and post-correction modules.

## Installation

```bash
git clone https://github.com/areopagusworkshop/msocr.git
cd msocr
uv sync
```

## Quick Start

### 1. Printed OCR (image)
```bash
uv run msocr ocr --lang latin test/benchmarks/printed/latin/latin_sample_001.png
```

### 2. Printed OCR (PDF)
```bash
uv run msocr ocr --lang latin /path/to/file.pdf
```

### 3. Handwritten route
```bash
uv run msocr htr --lang syriac --provider auto --variant default /path/to/manuscript_line.png
```

### 4. Benchmark (printed)
```bash
uv run msocr benchmark \
  --manifest test/benchmarks/manifests/printed_all.json \
  --output output/benchmarks/printed_all_report.json \
  --cer-threshold 0.05
```

### 5. Start API backend
```bash
uv run msocr api --host 127.0.0.1 --port 8000
```

### 6. Start API backend with a HAR-backed handwritten model
```bash
export MSOCR_HTR_RUNTIME_HAR_REGISTRY=msocr-models
export MSOCR_HTR_RUNTIME_HAR_VERSION=v42
export MSOCR_HTR_RUNTIME_HAR_FILENAME=model.mlmodel
export MSOCR_HTR_RUNTIME_LANG=syriac
export MSOCR_HTR_RUNTIME_VARIANT=default
export MSOCR_HTR_RUNTIME_HAR_CACHE_DIR=models/runtime/htr

uv run msocr api --host 127.0.0.1 --port 8000
```

### 7. Plan a RunPod training job
```bash
uv run msocr runpod-submit \
  --manifest-id syriac-printed-v1 \
  --lang syriac \
  --script-variant estrangela
```

### 8. Plan a HAR model publication
```bash
uv run msocr har-publish \
  --registry msocr-models \
  --lang syriac \
  --script-variant estrangela \
  --version v14 \
  --model-file models/finetune/paynesmith_serto_v1.mlmodel
```

For handwritten runtime artifacts, use `--writing-mode handwritten` and a `.mlmodel`:

```bash
uv run msocr har-publish \
  --registry msocr-models \
  --lang syriac \
  --script-variant default \
  --writing-mode handwritten \
  --version v42 \
  --model-file /path/to/model.mlmodel
```

### 9. Plan the full promotion workflow
```bash
uv run msocr pipeline-submit \
  --manifest-id syriac-train-v1 \
  --benchmark-manifest-id syriac-bench-v1 \
  --lang syriac \
  --script-variant estrangela \
  --model-file models/tesseract/syr_serto.traineddata \
  --registry msocr-models
```

`pipeline-submit` is currently complete for printed benchmark/gate/promotion flows. Handwritten runtime deployment is supported, but handwritten benchmark/promotion parity still uses dedicated Harness YAML examples.

### 10. Validate a live runtime with smoke checks
```bash
uv run msocr runtime-smoke-check \
  --mode handwritten \
  --lang syriac \
  --variant default \
  --provider auto \
  --base-url http://127.0.0.1:8000 \
  --image /path/to/manuscript_line.png \
  --require-engine kraken
```

### 11. Start Gradio demo
```bash
uv run msocr demo --host 127.0.0.1 --port 7860
```

With handwritten HAR runtime enabled, Gradio uses the same `MSOCR_HTR_RUNTIME_*` env vars as the API.

### 12. Use the Harness delegate examples

- Printed runtime delegate pipeline:
  - `pipeline/harness/syriac_printed_train_delegate.yaml`
- Syriac handwritten runtime + Gradio delegate pipeline:
  - `pipeline/harness/syriac_handwritten_runtime_delegate.yaml`

## CLI Reference

### `ocr`
```bash
uv run msocr ocr [OPTIONS] INPUT_PATH
```
- `INPUT_PATH`: required positional input (image or PDF)
- Key options:
  - `--lang` (required)
  - `--engine auto|kraken|tesseract`
  - `--model` (optional override)
  - `--syriac-variant default|estrangela|serto|east`
  - `--reference-text` and `--cer-threshold` (Syriac variant gating)

### `htr`
```bash
uv run msocr htr [OPTIONS] INPUT_PATH
```
- `INPUT_PATH`: required positional input (image or PDF)
- Key options:
  - `--lang` (required)
  - `--provider auto|kraken|transkribus`
  - `--variant <script_variant>`
  - `--model` (optional override)

For Syriac handwritten requests, `--provider auto` now means:
- use local Kraken inference when a HAR-backed handwritten runtime model is configured
- otherwise fall back to the existing Transkribus bridge response

### `train`
```bash
uv run msocr train --lang <lang> --mode ocr|htr [--config ...] [--gt-dir ...|--gt-file ...]
```

### `runpod-submit`
```bash
uv run msocr runpod-submit --manifest-id <split_manifest_id> --lang <lang> [OPTIONS]
```
- Builds a manifest-aware RunPod pod request using the official pods REST API shape.
- Defaults to dry-run JSON output; add `--execute` to submit and `--wait` to poll for `RUNNING`.

### `har-publish`
```bash
uv run msocr har-publish --registry <registry> --lang <lang> --version <version> --model-file <path> [OPTIONS]
```
- Builds a Harness Artifact Registry generic package publication plan.
- Defaults to dry-run JSON output; add `--execute` to invoke the `hc` CLI.

### `pipeline-submit`
```bash
uv run msocr pipeline-submit --manifest-id <train_manifest_id> --benchmark-manifest-id <benchmark_manifest_id> --lang <lang> [OPTIONS]
```
- Builds a single manifest-aware workflow covering RunPod submission, benchmark evaluation, policy gating, and HAR promotion.
- Defaults to dry-run JSON output; add `--execute` to submit the pod, run the benchmark, and publish only if the CER gate passes.

### `runtime-smoke-check`
```bash
uv run msocr runtime-smoke-check --mode printed|handwritten --lang <lang> [OPTIONS]
```
- Supports local resolution-only validation or live HTTP probing with `--base-url`.
- Printed mode can probe `/ocr` and uses the canonical sample in `assets/runtime/` when `--base-url` is set and `--image` is omitted.
- Handwritten mode probes `/htr` and requires a caller-supplied manuscript image via `--image`.
- Use `--require-engine` when you need to assert that the runtime is serving the promoted engine you expect.

## API Endpoints

When running `api`:
- `GET /health`
- `POST /ocr`
- `POST /htr`

`POST /htr` accepts either JSON or multipart form uploads and supports:
- `lang`
- `variant`
- `provider`
- `model` (optional explicit override)
- `device`

OpenAPI docs:
- `http://127.0.0.1:8000/docs`

## Benchmarks

Benchmark assets and manifests:
- `test/benchmarks/printed/...`
- `test/benchmarks/references/...`
- `test/benchmarks/manifests/...`

Report output examples:
- `output/benchmarks/printed_<language>_report.json`
- `output/benchmarks/printed_all_report.json`

Policy:
- Printed acceptance gate: `CER <= 0.05`
- `WER` tracked as secondary diagnostic metric
- Failed cases should be marked for manual review

Handwritten runtime deployment is implemented, but handwritten benchmark/promotion parity is still tracked separately from the printed benchmark workflow.

## HAR Runtime Conventions

Package naming follows:

```text
{lang}-{script_variant}-{writing_mode}
```

Examples:
- `syr-estrangela-printed`
- `syr-default-handwritten`

Printed runtime env vars:
- `MSOCR_PRINTED_RUNTIME_HAR_REGISTRY`
- `MSOCR_PRINTED_RUNTIME_HAR_VERSION`
- `MSOCR_PRINTED_RUNTIME_HAR_FILENAME`
- `MSOCR_PRINTED_RUNTIME_HAR_PACKAGE` (optional explicit package override)
- `MSOCR_PRINTED_RUNTIME_HAR_CACHE_DIR` (optional)

Handwritten runtime env vars:
- `MSOCR_HTR_RUNTIME_HAR_REGISTRY`
- `MSOCR_HTR_RUNTIME_HAR_VERSION`
- `MSOCR_HTR_RUNTIME_HAR_FILENAME`
- `MSOCR_HTR_RUNTIME_LANG`
- `MSOCR_HTR_RUNTIME_VARIANT`
- `MSOCR_HTR_RUNTIME_HAR_PACKAGE` (optional explicit package override)
- `MSOCR_HTR_RUNTIME_HAR_CACHE_DIR` (optional)

## Model Paths (Current Conventions)

### Kraken
- Greek printed primary:
   - `models/kraken/greek-english_porson_sophoclesplaysa05campgoog/...mlmodel`
- Greek fallback:
   - `models/kraken/greek-german_serifs_sophokle1v3soph/...mlmodel`
   - `models/kraken/greek-german_serifs_bsb10234118/...mlmodel`
- Latin printed primary (CATMuS-Print Large 2024-01-30):
   - `models/kraken/catmus-print-fondue-large.mlmodel`
   - DOI: 10.5281/zenodo.10592716
   - Accuracy: 98.56%, CER: 1.44%
- Latin handwritten (CATMuS Medieval):
   - `models/kraken/catmus-medieval-1.5.0.mlmodel`
   - DOI: 10.5281/zenodo.10066218
   - Supports: 8-15th century manuscripts

### Tesseract local traineddata
- Coptic:
  - `models/tesseract/cop.traineddata`
- Armenian:
  - `models/tesseract/hye-calfa-n.traineddata`
- Syriac custom variant slots (optional):
  - `models/tesseract/syr_serto.traineddata`
  - `models/tesseract/syr_east.traineddata`

## Project Layout (Current)

```text
msocr/
├── msocr/
│   ├── cli.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── pipeline/
│   ├── pipelines/
│   ├── preprocessing/
│   ├── service/
│   │   ├── api.py
│   │   └── gradio_demo.py
│   ├── training/
│   └── utils/
├── instruction/
├── pipeline/
├── skill/
├── source_registry/
├── test/benchmarks/
├── models/
├── output/
└── pyproject.toml
```

## Notes

- `models/` and `output/` are runtime artifact directories and are gitignored.
- OCRopus/Ocropy fallback is deactivated in the current implementation phase.
- Default training method for Syriac Payne-Smith pipeline is RunPod (persistent pod), see `pipeline/runpod_train_reference.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, conventions, and pull request guidelines.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
