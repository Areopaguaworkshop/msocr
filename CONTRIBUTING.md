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
uv run pytest tests/service/
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
├── language_registry.py   # Canonical Sogdian profile & normalize
├── models/
│   └── inference.py       # Kraken HTR inference wrapper
├── service/
│   ├── api.py             # FastAPI endpoints (/health, /htr)
│   ├── gradio_demo.py     # Gradio browser UI
│   ├── runtime.py         # HTR service dispatcher
│   ├── deploy.py          # Server startup + runtime smoke checks
│   └── annotation_api.py  # Annotation session API (port 8001)
├── evaluation/
│   └── metrics.py         # CER/WER computation helpers
├── training/
│   └── ketos_trainer.py
├── data/
│   └── manifest.py         # Frozen split manifest loading
└── output/
    └── formats.py          # JSON/Markdown output writers
```

## Key Conventions

### HTR-Only Scope

The active project is Sogdian manuscript HTR with remote training. RunPod GPU Cloud Pod submission is supported for `ketos train` fine-tuning via `msocr train-remote`. Multi-stage orchestration is a minimal procedural walker (one style-group at a time), not a DAG engine. Tesseract/OCRmyPDF/printed-OCR/HAR remain out of scope.

### Language Codes

CLI accepts `sogdian` plus alias `old_sogdian`. Use `normalize_language_code()` from `language_registry.py` for resolution.

### Runtime Model Resolution

API, CLI, and Gradio resolve local Kraken models from explicit `--model`/request fields, `MSOCR_HTR_RUNTIME_MODEL_PATH`, `MSOCR_HTR_MODEL_PATH`, `MSOCR_RUNTIME_MODEL_PATH`, then `models/kraken/sogdian_manuscript.mlmodel`.

### Segmentation Mode

Sogdian manuscript recognition uses RTL reading order. Prefer Kraken `horizontal-rl` behavior for segmentation and recognition.

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
- Default Sogdian runtime model convention → `models/kraken/sogdian_manuscript.mlmodel`

## Manifest Policy

- Manifests live in `data/manifests/` using `.json` format
- Active manifests must use `language: "sogdian"` and `writing_mode: "handwritten"`
- Split manifests must isolate `manuscript_id` across partitions

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
