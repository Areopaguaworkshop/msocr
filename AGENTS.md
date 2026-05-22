# AGENTS.md — msocr

Manuscript OCR/HTR toolkit. Kraken + Tesseract engines, route-aware language handling, benchmark/evaluation, RunPod training orchestration, HAR model promotion, FastAPI service, and Gradio demo.

## Quick Start

```bash
uv sync                # install dependencies (requires Python 3.12 per .python-version)
uv run msocr ocr --lang latin test/benchmarks/printed/latin/latin_sample_001.png
uv run msocr api --host 127.0.0.1 --port 8000
uv run msocr demo --host 127.0.0.1 --port 7860
```

CLI entrypoint: `msocr.cli:main` (registered as `msocr` in pyproject.toml).

## Architecture

Single Python package (`msocr/`) with a Click CLI, FastAPI service, and Gradio demo. No monorepo, no sub-packages.

```
msocr/
├── cli.py              # All CLI subcommands (click group)
├── language_registry.py # Canonical language profiles & normalize
├── pipelines/
│   ├── printed_ocr.py  # Printed OCR routing (Kraken↔Tesseract)
│   └── payne_smith.py  # Payne-Smith Syriac training pipeline
├── models/
│   └── inference.py    # Kraken/Tesseract inference wrapper
├── service/
│   ├── api.py          # FastAPI endpoints (/health, /ocr, /htr)
│   ├── gradio_demo.py  # Gradio browser UI
│   ├── runtime.py      # HTR service dispatcher
│   ├── deploy.py       # Server startup + runtime smoke checks
│   └── annotation_api.py  # Annotation session API (port 8001)
├── evaluation/
│   ├── metrics.py      # CER/WER computation
│   └── printed_benchmark.py
├── pipeline/
│   ├── runpod_client.py # RunPod API client
│   ├── har_client.py   # Harness Artifact Registry client
│   └── workflow.py     # End-to-end training→benchmark→promotion
├── training/
│   └── ketos_trainer.py
├── data/
│   └── manifest.py     # Frozen split manifest loading
└── output/
    └── formats.py      # JSON/Markdown/PDF output writers
```

## Key Conventions

- **Route separation is mandatory**: All code and tests must keep `printed` (OCR) and `handwritten` (HTR) routes strictly separated. The `writing_mode` field (`printed`|`handwritten`) drives routing everywhere — CLI, API, runtime resolution, and benchmarks.
- **Language codes**: CLI accepts long names (`greek`, `latin`, `syriac`, `coptic`, `armenian`, `geez`, `sogdian`, `old_turkish`) plus alias `armenia`→`armenian`. `normalize_language_code()` in `language_registry.py` resolves them.
- **Syriac variants**: Printed Syriac has CER-gated fallback logic. `default` uses Tesseract `syr`, then CER-gated switch to `serto`/`east` traineddata when available and reference text shows the threshold is met.
- **HAR runtime model resolution**: API and Gradio startup resolve models from Harness Artifact Registry via `MSOCR_*_HAR_*` env vars. Printed and handwritten paths use separate env var sets (see README "HAR Runtime Conventions").

## CLI Subcommands

| Command | Purpose |
|---|---|
| `ocr` | Printed OCR (image/PDF) |
| `htr` | Handwritten HTR (image/PDF) |
| `train` | Kraken ketos training |
| `preprocess` | Image preprocessing |
| `benchmark` | Printed benchmark → JSON report |
| `runpod-submit` | Create/submit RunPod training job |
| `har-publish` | Publish model to Harness Artifact Registry |
| `pipeline-submit` | End-to-end train→benchmark→promote workflow |
| `runtime-smoke-check` | Validate runtime model resolution (local or HTTP) |
| `api` | FastAPI backend on port 8000 |
| `annotation-api` | Annotation session API on port 8001 |
| `demo` | Gradio browser UI on port 7860 |
| `payne-smith` | Payne-Smith Syriac training pipeline |

**Important**: `runpod-submit`, `har-publish`, and `pipeline-submit` default to **dry-run** (print plan JSON only). Must pass `--execute` to actually submit/publish.

## Testing

```bash
uv run pytest              # run all tests
uv run pytest tests/evaluation/  # run a single test directory
```

Tests use `pytest` with `monkeypatch` and `tmp_path` fixtures. No special services required to run unit tests (model inference is mocked). The `test/` directory at repo root is gitignored — canonical test fixtures live under `tests/`.

## Models Directory

`models/` is **gitignored** (large binary files). Only placeholder `.gitkeep` files and `models/README.md` are tracked. Models must be downloaded or resolved separately:
- Kraken `.mlmodel` files go under `models/kraken/`
- Tesseract `.traineddata` files go under `models/tesseract/`
- HAR runtime models cache to `models/runtime/` (printed) or `models/runtime/htr/` (handwritten)

Key model paths referenced in code:
- `models/kraken/catmus-print-fondue-large.mlmodel` — Latin printed primary
- `models/kraken/catmus-medieval-1.5.0.mlmodel` — Latin handwritten
- `models/kraken/greek-english_porson_sophoclesplaysa05campgoog/` — Greek printed primary
- `models/tesseract/cop.traineddata` — Coptic

## Benchmarks

- Manifests live in `data/manifests/` and use `.json` format with `manifest_id`, `writing_mode`, `language`, and `partitions` (train/validation/holdout keyed by `manuscript_id`).
- Acceptance gate: printed `CER ≤ 0.05`, handwritten `CER ≤ 0.10`.
- `test/benchmarks/` at repo root is gitignored. Benchmark references and manifests under it are not in git.

## Environment Variables

| Prefix | Scope |
|---|---|
| `MSOCR_PRINTED_RUNTIME_HAR_*` | Printed model HAR resolution (REGISTRY, VERSION, FILENAME, PACKAGE, CACHE_DIR) |
| `MSOCR_HTR_RUNTIME_HAR_*` | Handwritten model HAR resolution |
| `MSOCR_HTR_RUNTIME_LANG` | Handwritten runtime language |
| `MSOCR_HTR_RUNTIME_VARIANT` | Handwritten runtime variant |
| `MSOCR_NOTIFICATION_URL` | Pipeline webhook notification target |
| `RUNPOD_SSH_KEY_PATH` | SSH key for RunPod model retrieval |
| `RUNPOD_SSH_PUBLIC_KEY` | Public key injected into RunPod pod |

## Python Version

`.python-version` specifies `3.12`. `pyproject.toml` states `>=3.9` but the project targets 3.12 in practice.

## Build & Deploy

Docker build: `Dockerfile` uses `python:3.12-slim`, runs `uv sync --frozen`, copies `msocr/`, `models/`, and `source_registry/`. Default CMD is the API server.

## Gotchas

- The segmentation mode for Kraken models matters: CATMuS models require **baseline** segmentation (`BaselineLine`), not bounding-box (`BBoxLine`). The code in `msocr/models/inference.py` handles this — don't revert to BBoxLine.
- `--syriac-variant` only affects printed OCR routing. Handwritten Syriac uses `--variant` (a different flag).
- PDF output is not supported for `htr` (only JSON/Markdown).
- The `payne-smith` subcommand has a `--phases` flag with a very long default value listing all phases. It defaults to dry-run unless `--execute` is passed.
- `configs/` directory under `msocr/` is currently empty; training configs `sogdian_config.yaml` and `old_turkish_config.yaml` are referenced from `DEFAULT_CONFIGS` in cli.py but don't exist in the checked-out tree — they may need to be created or the references updated.