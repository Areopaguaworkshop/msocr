# Agent Model Blueprint: Patristic OCR/HTR for `msocr`

## 1. Objective
Define a conservative, implementation-ready agent model for manuscript OCR/HTR that:
- preserves current `msocr` CLI workflow,
- separates printed and handwritten routes from the beginning,
- starts implementation with Greek/Latin printed OCR, then expands to other patristic languages.

## 1.1 Implementation Constraints
- All code changes must be under `msocr/`.
- Implementation should follow structured module boundaries (`data`, `preprocessing`, `models`, `training`, `utils`).
- Python workflows should use `uv`.

## 2. Runtime Contract
### Required runtime inputs
- `target_language`: `sogdian | old_turkish | greek | latin | syriac | coptic`
- `writing_mode`: `printed | handwritten`
- `scan_dpi`: integer
- `desired_accuracy`: `baseline | research | critical`

### Optional runtime inputs
- `century`: integer or range
- `compute_budget`: `low | medium | high`
- `normalization_profile`: language-specific normalization alias

### Hard constraints
- Route must be chosen by `writing_mode` at initialization.
- Fallback path is required when confidence/routing stability is low.
- Output must include provenance and review flags.

## 3. Route Definitions
### Printed OCR route
1. `preprocess` profile tuned for printed pages
2. script/direction validation
3. printed model selection
4. OCR inference
5. post-correction (lexicon + normalization)
6. evaluation gate (`CER` primary, `WER` secondary)

### Handwritten HTR route
1. `preprocess` profile tuned for manuscripts/handwriting
2. script/direction validation
3. handwritten model selection
4. HTR inference
5. post-correction (lexicon + normalization)
6. evaluation gate (`CER` primary, `WER` secondary)

## 4. Fallback and Manual Review
- Trigger fallback if classifier confidence is low or model scores are unstable.
- Fallback action: dual-model sampling or conservative rerun with stronger preprocessing.
- Emit `needs_manual_review: true` if either acceptance metric fails.

## 5. Accuracy Policy
- Printed target: `CER <= 5%`
- Handwritten target: `CER <= 10%`
- `WER` is mandatory for diagnosis but not the primary pass/fail gate.
- For languages lacking strong public handwritten models, require custom HTR training.

## 6. Data and Benchmark Policy
- Use predefined, frozen split manifests.
- Split by `manuscript_id` to prevent leakage across train/val/test.
- Store benchmark results with model/version metadata for reproducibility.

## 7. `msocr` Integration Map
### Implemented modules
- `msocr/cli.py`
- `msocr/data/manager.py`
- `msocr/data/annotation.py`
- `msocr/preprocessing/pipeline.py`
- `msocr/models/inference.py`
- `msocr/training/ketos_trainer.py`

### Planned/external modules
- script-mode classifier
- language-aware router
- lexicon+morphology correction
- connectors: eScriptorium/Transkribus

## 8. Language Availability Matrix
### Fully specified now
- Sogdian: printed + handwritten training/evaluation profile

### Partially available
- Old Turkish: implemented config path exists; full profile alignment pending

### Planned
- Greek, Latin, Syriac, Coptic (with explicit handwritten training requirement where needed)

## 9. Output Contract
Each run should emit structured fields:
- `raw_text`
- `normalized_text`
- `corrected_text`
- `metrics`: `CER`, `WER`
- `route`: printed/handwritten
- `model_id` and `model_version`
- `needs_manual_review`
- `provenance` (data split id, preprocessing profile, timestamp)

## 10. Extension Strategy
- Add language profiles without breaking current CLI order.
- Keep engine abstraction stable to allow backend swap.
- Promote planned profiles to implemented only after repeatable benchmark validation.
