# Strat-here: msocr Instruction Baseline

## Purpose
- This folder defines the general documentation-first agent specification for `msocr`.
- You can write implementation code now, in addition to agent mode Markdown and YAML.
- All implementation code must be placed under the `msocr/` folder with a structured module layout.
- Python environment, dependency, and command workflows should use `uv`.

## Project Context Snapshot
- Repository: `msocr`
- Current focus in codebase: Sogdian and Old Turkish manuscript OCR using Kraken (`ketos`)
- Existing pipeline components:
  - CLI: `ocr`, `preprocess`, `train`
  - Data manager for datasets and metadata
  - Annotation export/import (Label Studio, CVAT, ALTO/PAGE)
  - Image preprocessing pipeline
  - Kraken inference wrapper
  - Kraken ketos training wrapper
- Existing training configs:
  - `configs/sogdian_config.yaml` (LTR)
  - `configs/old_turkish_config.yaml` (RTL)

## Required Documentation Outputs
1. `summary.md`
2. agent.patristic.ocr.model.md in this folder
3. skill yamls in skill folder 
4. pipeline yamls in pipeline folder 
5. source registry yaml in source_registry folder
6. state_update yaml in output folder 

## Authoring Rules
- Markdown and YAML remain required for specs, and code implementation is now allowed.
- Keep all new code under `msocr/` and avoid scattering implementation logic outside the package.
- Use `uv` for Python package and runtime workflows.
- Keep claims conservative and implementation-realistic.
- Separate:
  - printed OCR routing
  - handwritten HTR routing
- Include explicit fallback behavior for low-confidence classification.
- Keep model availability claims explicit:
  - if no strong public handwritten model exists, state that custom HTR training is required.

## Integration Rules for msocr
- Map proposed architecture to current `msocr` modules where possible.
- Mark items not yet implemented in code as "planned" or "external".
- Preserve compatibility with current CLI flow:
  - preprocess -> annotation -> train -> ocr
- Add extension strategy for Greek, Latin, Syriac, and Coptic without claiming those are already implemented.

## Output Tone
- Technical, concise, and implementation-ready.
- No marketing language.
- No exaggerated accuracy claims.

### What Is Done?
- Core CLI structure is implemented with subcommands for:
  - `ocr` (printed)
  - `htr` (handwritten)
  - `train`
  - `preprocess`
- Greek and Latin printed OCR routing is implemented.
  - Greek printed: Kraken primary + fallback models
  - Latin printed: Kraken CATMuS-Print Large with Tesseract fallback
- Greek and Latin handwritten defaults are wired in current runtime.
- Syriac printed OCR baseline is implemented with Tesseract (`syr`) and variant-aware logic.
- Coptic printed OCR is implemented with Tesseract (`cop`) route.
- Armenian and Geez printed OCR routes are implemented with Tesseract.
  - Armenian prefers local `hye-calfa-n` when available
- OCRopus/Ocropy fallback is deactivated in current phase.
- Runtime model/output artifact policy is set to ignore generated assets in git.

## Next Step
- Keep printed OCR and handwritten HTR routes strictly separated in all new modules and tests.
- Complete benchmark validation on real datasets (CER/WER) for implemented printed routes.
  - target: printed CER <= 5%
- Finish Syriac handwritten production path:
  - use Transkribus platform for initial recognition
  - export PAGE XML or ALTO XML
  - train local Kraken Syriac HTR model from exported XML + images
- Finish Coptic handwritten production path:
  - use Transkribus workflow for dataset generation
  - export PAGE XML or ALTO XML with corrected transcriptions
  - train Kraken HTR model; keep Tesseract as backup training/OCR option
- Implement Armenian handwritten HTR path:
  - preprocessing (cleanup, deskew, crop)
  - layout analysis (eScriptorium or Transkribus)
  - Kraken training from PAGE/ALTO XML
- Implement Geez handwritten HTR path with the same preparation and training sequence as Armenian.
- Align and test Sogdian and Old Turkish handwritten-focused pipeline under the same benchmark/reporting standard.
