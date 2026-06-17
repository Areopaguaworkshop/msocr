# msocr Documentation

`msocr` is a CLI project for manuscript OCR only, right now it only focused on Sogdian manuscript HTR. 

## plan
### For each manuscripts, depend their time zone, alphabit and scriber's writing sytle, you should training a separate models for this through Kraken training in RunPod platform. 
### I will begin with Christian text Berlin Turfan, training a Jingjiao Sogdian manuscript OCR models, Then build a benchmark and evaluate system for each training model. 
### the pipeline/workflow is like this, input is image or pdf, then it will go through line segmentation, extract the lines in manuscript line by line, then put it on Astro web UI, also line by line, one line image then one line transcription below, then output as dataset for training model through Kraken in RunPod platfrom, then it will training the models in RunPod, after the model trained. It will go through a evaluate processing to evaluate its CER and so on for standard ways. 
Note: You should check Kraken, RunPod in details so that I do a details plan for this project, by the way, you should also make you plan as html file where Can read through Astro web UI. 

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

Removed historical scope: printed OCR routing, Tesseract/OCRmyPDF fallbacks, RunPod submission, HAR promotion, and multi-stage orchestration.
