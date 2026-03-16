# msocr

Manuscript OCR/HTR toolkit with route-aware language handling, Kraken/Tesseract integration, benchmark reporting, FastAPI service APIs, and Gradio demo UI.

## Current Status

### Implemented
- CLI with subcommands:
  - `ocr` (printed OCR, image/PDF input)
  - `htr` (handwritten OCR/HTR, image/PDF input)
  - `train` (Kraken ketos training)
  - `preprocess`
  - `benchmark`
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
   - Syriac: Transkribus workflow bridge message path
- Printed benchmark runner with CER/WER JSON reports
- FastAPI backend (`msocr/service/api.py`)
- Gradio browser demo (`msocr/service/gradio_demo.py`)

### Planned
- Production handwritten workflows for Syriac/Coptic/Armenian/Geez with local Kraken training from PAGE/ALTO exports.
- Expanded language-aware classifier/router and post-correction modules.

## Installation

```bash
git clone <repository-url>
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
uv run msocr htr --lang latin /path/to/line_or_page.png
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

### 6. Start Gradio demo
```bash
uv run msocr demo --host 127.0.0.1 --port 7860
```

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
  - `--model` (optional override)

### `train`
```bash
uv run msocr train --lang <lang> --mode ocr|htr [--config ...] [--gt-dir ...|--gt-file ...]
```

## API Endpoints

When running `api`:
- `GET /health`
- `POST /ocr`
- `POST /htr`

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
