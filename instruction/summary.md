# Manuscript OCR System for Church Fathers (`msocr`)

## 1. Purpose and Scope
- Build a documentation-first OCR/HTR blueprint for historical Church Fathers corpora.
- Support both printed OCR and handwritten HTR, with strict route separation.
- Keep current CLI-compatible flow: `preprocess -> annotation -> train -> ocr`.
- Start with Sogdian as the fully specified training profile; expand to other languages incrementally.

## 2. Supported Language Status
### Implemented in current codebase (training/inference path exists)
- Sogdian (`LTR`): `configs/sogdian_config.yaml`
- Old Turkish (`RTL`): `configs/old_turkish_config.yaml`

### Planned (not fully implemented in code)
- Greek (polytonic)
- Latin (historical)
- Syriac (Estrangelo/Serto/East Syriac)
- Coptic (Sahidic/Bohairic)

### External/optional integrations
- eScriptorium
- Transkribus
- morphology and lexicon APIs

## 3. Route Separation (Required)
### Printed OCR Route
- Use printed-oriented preprocessing and model profiles.
- Primary target metric: `CER <= 5%`.
- Secondary metric: `WER` for diagnostic quality tracking.

### Handwritten HTR Route
- Use handwriting-oriented preprocessing and model profiles.
- Primary target metric: `CER <= 10%`.
- Secondary metric: `WER` for diagnostic quality tracking.
- For languages without strong public handwritten models, custom HTR training is required.

## 4. Routing Rules
- `writing_mode` must be declared at run design time (`printed` or `handwritten`).
- If `writing_mode=auto` is used for exploration, output must include explicit confidence and may trigger manual review.
- If low confidence or unstable model selection occurs, run fallback and emit `needs_manual_review: true`.
- If direction is RTL (for example Old Turkish, Syriac), enforce RTL normalization and output handling.

## 5. Benchmark and Evaluation Policy
- Metrics: both `CER` and `WER` are required.
- Acceptance gate: `CER` is primary; `WER` is secondary but tracked.
- Split policy: predefined frozen manifests grouped by `manuscript_id` to avoid data leakage.
- Manual review trigger: if either required metric gate fails for the selected profile.

## 6. Training Data Policy
- Minimum useful fine-tuning: 500-1,000 aligned lines.
- Strong baseline: 2,000-5,000 aligned lines.
- Emphasize line-level pairs for recognition model training.
- Keep page-level annotations for layout/segmentation tasks.

## 7. Integration Mapping to Existing `msocr` Modules
### Implemented components
- CLI orchestration: `msocr/cli.py`
- Dataset metadata and storage: `msocr/data/manager.py`
- Annotation bridge: `msocr/data/annotation.py`
- Preprocessing pipeline: `msocr/preprocessing/pipeline.py`
- Inference wrapper: `msocr/models/inference.py`
- Kraken training wrapper: `msocr/training/ketos_trainer.py`

### Planned additions
- script/language + mode classifier layer
- language-aware engine router
- lexicon/morphology post-correction module
- expanded language profiles (Greek/Latin/Syriac/Coptic)

## 8. Deliverables in This Documentation Baseline
- `instruction/summary.md` (this file)
- `instruction/agent.patristic.ocr.model.md`
- `instruction/eval.md`
- `instruction/code_review.md`
- skill YAMLs in `skill/`
- pipeline YAMLs in `pipeline/`
- source registry YAML in `source_registry/`
- state update YAML in `output/`

## 9. Maintenance Policy
- This file is a full rewrite baseline for the current planning cycle.
- Later updates should be incremental and tied to actual project progress.
- Do not promote planned/external items to implemented until code paths exist and are validated.
