# Contributing to msocr

Thank you for your interest in contributing! This guide covers setup, development conventions, and how to submit changes.

## Quick Start

```bash
# Clone and install
git clone https://github.com/areopagusworkshop/msocr.git
cd msocr
uv sync

# Run tests
uv run pytest

# Run a single test directory
uv run pytest tests/evaluation/
```

Requires Python 3.12 (see `.python-version`).

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/) for dependency management
2. Clone the repository
3. Run `uv sync` to create a virtual environment and install dependencies
4. Download model files (see [Models Directory](#models-directory) below)
5. Run `uv run msocr --help` to verify the CLI works

## Architecture

Single Python package (`msocr/`) with a Click CLI, FastAPI service, and Gradio demo. No monorepo, no sub-packages.

```
msocr/
├── cli.py                # All CLI subcommands (click group)
├── language_registry.py   # Canonical language profiles & normalize
├── pipelines/
│   ├── printed_ocr.py    # Printed OCR routing (Kraken↔Tesseract)
│   └── payne_smith.py    # Payne-Smith Syriac training pipeline
├── models/
│   └── inference.py       # Kraken/Tesseract inference wrapper
├── service/
│   ├── api.py             # FastAPI endpoints (/health, /ocr, /htr)
│   ├── gradio_demo.py     # Gradio browser UI
│   ├── runtime.py         # HTR service dispatcher
│   ├── deploy.py          # Server startup + runtime smoke checks
│   └── annotation_api.py  # Annotation session API (port 8001)
├── evaluation/
│   ├── metrics.py         # CER/WER computation
│   └── printed_benchmark.py
├── pipeline/
│   ├── runpod_client.py    # RunPod API client
│   ├── har_client.py       # Harness Artifact Registry client
│   └── workflow.py         # End-to-end training→benchmark→promotion
├── training/
│   └── ketos_trainer.py
├── data/
│   └── manifest.py         # Frozen split manifest loading
└── output/
    └── formats.py          # JSON/Markdown/PDF output writers
```

## Key Conventions

### Route Separation (Mandatory)

All code and tests **must** keep `printed` (OCR) and `handwritten` (HTR) routes strictly separated. The `writing_mode` field (`printed`|`handwritten`) drives routing everywhere — CLI, API, runtime resolution, and benchmarks.

### Language Codes

CLI accepts long names (`greek`, `latin`, `syriac`, `coptic`, `armenian`, `geez`, `sogdian`, `old_turkish`) plus alias `armenia`→`armenian`. Use `normalize_language_code()` from `language_registry.py` for resolution.

### Syriac Variants

Printed Syriac has CER-gated fallback logic. The `default` variant uses Tesseract `syr`, then CER-gated switch to `serto`/`east` traineddata when available. **Do not** modify this routing without updating the benchmark thresholds.

### HAR Runtime Model Resolution

API and Gradio startup resolve models from Harness Artifact Registry via `MSOCR_*_HAR_*` env vars. Printed and handwritten paths use separate env var sets.

### Segmentation Mode

CATMuS models require **baseline** segmentation (`BaselineLine`), not bounding-box (`BBoxLine`). The code in `msocr/models/inference.py` handles this — do not revert to BBoxLine.

### CLI Flag Distinction

- `--syriac-variant` only affects **printed** OCR routing
- `--variant` is used for **handwritten** runtime model selection
- These are different flags with different scopes

### Dry-Run Defaults

`runpod-submit`, `har-publish`, and `pipeline-submit` default to **dry-run** (print plan JSON only). Users must pass `--execute` to actually submit/publish. New commands that submit external resources should follow this pattern.

## Testing

```bash
# Run all tests
uv run pytest

# Run a specific test directory
uv run pytest tests/evaluation/

# Run with verbose output
uv run pytest -v
```

Tests use `pytest` with `monkeypatch` and `tmp_path` fixtures. No special services are required to run unit tests — model inference is mocked.

**Test file location**: `tests/` (not `test/`, which is gitignored for runtime artifacts).

## Models Directory

The `models/` directory is **gitignored** (large binary files). Only placeholder `.gitkeep` files and `models/README.md` are tracked. Models must be downloaded or resolved separately:

- Kraken `.mlmodel` files → `models/kraken/`
- Tesseract `.traineddata` files → `models/tesseract/`
- HAR runtime models cache → `models/runtime/` (printed) or `models/runtime/htr/` (handwritten)

## Benchmark Policy

- Printed acceptance gate: **CER ≤ 0.05**
- Handwritten acceptance gate: **CER ≤ 0.10**
- Manifests live in `data/manifests/` using `.json` format

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear, descriptive commit messages
4. Add or update tests as needed
5. Run `uv run pytest` to verify all tests pass
6. Ensure code follows the project conventions above
7. Submit a pull request with a description of the changes

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce, expected behavior, and actual behavior
- Specify the msocr version (`uv run msocr --version` or check `pyproject.toml`)
- Include relevant language, engine, and model information

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).