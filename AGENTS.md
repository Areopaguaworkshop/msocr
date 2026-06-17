# AGENTS.md — msocr

Focused Sogdian manuscript HTR toolkit. Runtime recognition is local Kraken HTR only, with small supporting tools for preprocessing, ground-truth annotation, local training, a FastAPI service, and a Gradio demo.

## Quick Start

```bash
uv sync                # install dependencies (requires Python 3.12 per .python-version)
uv run msocr htr --lang sogdian --model models/kraken/sogdian_manuscript.mlmodel /path/to/page.png
uv run msocr api --host 127.0.0.1 --port 8000
uv run msocr demo --host 127.0.0.1 --port 7860
```

CLI entrypoint: `msocr.cli:main` (registered as `msocr` in pyproject.toml).

## Architecture

Single Python package (`msocr/`) with a Click CLI, FastAPI service, and Gradio demo. No monorepo, no sub-packages.

```
msocr/
├── cli.py              # All CLI subcommands (click group)
├── language_registry.py # Canonical Sogdian profile & normalize
├── models/
│   └── inference.py    # Kraken HTR inference wrapper
├── service/
│   ├── api.py          # FastAPI endpoints (/health, /htr)
│   ├── gradio_demo.py  # Gradio browser UI
│   ├── runtime.py      # HTR service dispatcher
│   ├── deploy.py       # Server startup + runtime smoke checks
│   └── annotation_api.py  # Annotation session API (port 8001)
├── evaluation/
│   └── metrics.py      # CER/WER computation helpers
├── training/
│   └── ketos_trainer.py
├── data/
│   └── manifest.py     # Frozen split manifest loading
└── output/
    └── formats.py      # JSON/Markdown output writers
```

## Key Conventions

- **HTR scope with remote training**: RunPod GPU Cloud Pod submission is supported for `ketos train` fine-tuning via `msocr train-remote`. Multi-stage orchestration is a minimal procedural walker (one style-group at a time), not a DAG engine. Tesseract/OCRmyPDF/printed-OCR/HAR remain out of scope.
- **Language codes**: CLI accepts `sogdian` plus alias `old_sogdian`. `normalize_language_code()` in `language_registry.py` resolves them.
- **Direction**: Sogdian manuscript handling is RTL. Kraken segmentation and annotation previews should use RTL/horizontal-rl behavior.
- **Runtime model resolution**: API, CLI, and Gradio resolve local Kraken models from explicit `--model`/request fields, `MSOCR_HTR_RUNTIME_MODEL_PATH`, `MSOCR_HTR_MODEL_PATH`, `MSOCR_RUNTIME_MODEL_PATH`, then `models/kraken/sogdian_manuscript.mlmodel`.

## CLI Subcommands

| Command | Purpose |
|---|---|
| `htr` | Sogdian manuscript HTR (image/PDF) |
| `train` | Kraken ketos training |
| `preprocess` | Image preprocessing |
| `runtime-smoke-check` | Validate runtime model resolution (local or HTTP) |
| `api` | FastAPI backend on port 8000 |
| `annotation-api` | Annotation session API on port 8001 |
| `demo` | Gradio browser UI on port 7860 |

## Testing

```bash
uv run pytest              # run all tests
uv run pytest tests/evaluation/  # run a single test directory
```

Tests use `pytest` with `monkeypatch` and `tmp_path` fixtures. No special services required to run unit tests (model inference is mocked). The `test/` directory at repo root is gitignored — canonical test fixtures live under `tests/`.

## Models Directory

`models/` is **gitignored** (large binary files). Only placeholder `.gitkeep` files and `models/README.md` are tracked. Models must be downloaded or resolved separately:
- Kraken `.mlmodel` files go under `models/kraken/`

Key model paths referenced in code:
- `models/kraken/sogdian_manuscript.mlmodel` — default Sogdian manuscript HTR model

## Benchmarks

- Manifests live in `data/manifests/` and use `.json` format with `manifest_id`, `writing_mode`, `language`, and `partitions` (train/validation/holdout keyed by `manuscript_id`).
- Only `language: "sogdian"` and `writing_mode: "handwritten"` are active.
- `test/benchmarks/` at repo root is gitignored. Benchmark references and manifests under it are not in git.

## Environment Variables

| Name | Scope |
|---|---|
| `MSOCR_HTR_RUNTIME_MODEL_PATH` | Preferred local Kraken model path |
| `MSOCR_HTR_MODEL_PATH` | Secondary local Kraken model path |
| `MSOCR_RUNTIME_MODEL_PATH` | Generic local Kraken model path |

## Python Version

`.python-version` specifies `3.12`. `pyproject.toml` states `>=3.9` but the project targets 3.12 in practice.

## Build & Deploy

Docker build: `Dockerfile` uses `python:3.12-slim`, runs `uv sync --frozen`, copies `msocr/` and `models/`. Default CMD is the API server.

## Gotchas

- Kraken baseline segmentation and Sogdian RTL direction matter: use `horizontal-rl` for segmented page/line recognition.
- Output is JSON or Markdown only. Searchable PDF output is intentionally not supported.
- PDF input is allowed only as a convenience; it is rendered to temporary page images before Kraken HTR.
