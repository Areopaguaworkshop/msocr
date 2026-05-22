# msocr Documentation

## Getting Started

- [README](../README.md) — Installation, quick start, CLI reference
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Development setup, conventions, PR process

## Architecture

- [AGENTS.md](../AGENTS.md) — Full architecture reference, conventions, gotchas
- `msocr/language_registry.py` — Canonical language profiles and normalization
- `msocr/cli.py` — All CLI subcommands (Click group)
- `msocr/service/api.py` — FastAPI endpoints (`/health`, `/ocr`, `/htr`)
- `msocr/service/runtime.py` — Runtime model resolution (HAR, local paths)

## Key Workflows

### Printed OCR

Input image/PDF → language routing → Kraken/Tesseract/OCRmyPDF → JSON/Markdown/PDF output

### Handwritten HTR

Input image/PDF → Kraken baseline segmentation → text → JSON/Markdown output

### Training Pipeline

RunPod submit → train → retrieve model → benchmark → CER gate → HAR promote → notify

### Benchmark

Manifest → run OCR per case → compare to reference → CER/WER metrics → pass/fail report

## Environment Variables

| Prefix | Scope |
|---|---|
| `MSOCR_PRINTED_RUNTIME_HAR_*` | Printed model HAR resolution |
| `MSOCR_HTR_RUNTIME_HAR_*` | Handwritten model HAR resolution |
| `MSOCR_HTR_RUNTIME_LANG` | Handwritten runtime language |
| `MSOCR_HTR_RUNTIME_VARIANT` | Handwritten runtime variant |
| `MSOCR_NOTIFICATION_URL` | Pipeline webhook notification target |
| `RUNPOD_SSH_KEY_PATH` | SSH key for RunPod model retrieval |
| `RUNPOD_SSH_PUBLIC_KEY` | Public key injected into RunPod pod |

## Project Plans

- `docs/plans/2026-04-04-msocr-instruction-implementation.md` — Implementation plan
- `docs/plans/2026-04-07-harness-os-progress.yaml` — Progress tracker