# msocr Documentation

`msocr` is now a focused Sogdian manuscript HTR toolkit. The supported runtime path is:

```text
manuscript image/PDF → Kraken segmentation → Sogdian HTR model → JSON/Markdown text
```

## Start here

- [README](../README.md) — installation, quick start, CLI/API/demo usage
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development setup and contribution notes

## Active architecture

- `msocr/language_registry.py` — canonical Sogdian profile and aliases
- `msocr/cli.py` — HTR, training, preprocessing, API, demo, and annotation commands
- `msocr/service/runtime.py` — local Kraken runtime model resolution
- `msocr/service/api.py` — FastAPI endpoints (`/health`, `/htr`)
- `msocr/service/gradio_demo.py` — browser demo for Sogdian HTR
- `msocr/training/ketos_trainer.py` — local Kraken `ketos` training wrapper
- `msocr/data/manifest.py` — frozen Sogdian split manifest loader

Removed historical scope: printed OCR routing, Tesseract/OCRmyPDF fallbacks, RunPod submission, HAR promotion, and pipeline/workflow orchestration.
