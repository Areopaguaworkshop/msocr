# Evaluation Standard (`eval.md`)

## 1. Purpose
Define a reproducible benchmark protocol for OCR/HTR model evaluation in `msocr`.

## 2. Metrics
- `CER` (Character Error Rate): edit distance at character level divided by total reference characters.
- `WER` (Word Error Rate): edit distance at token/word level divided by total reference words.

## 3. Metric Roles
- Primary acceptance metric: `CER`.
- Secondary diagnostic metric: `WER`.
- Both must be reported for every benchmark run.

## 4. Acceptance Gates
- Printed route pass target: `CER <= 5%`.
- Handwritten route pass target: `CER <= 10%`.
- Manual review trigger: if either required quality gate fails in the run policy.

## 5. Dataset Split Policy
- Use predefined frozen split manifests.
- Split unit: `manuscript_id`.
- No overlap of manuscript IDs across train/val/test.
- Any split revision must produce a new split version ID.

## 6. Benchmark Run Metadata (Mandatory)
Each run record must include:
- `benchmark_id`
- `language`
- `writing_mode`
- `model_id`
- `model_version`
- `preprocessing_profile`
- `split_version`
- `cer`
- `wer`
- `pass_fail`
- `needs_manual_review`

## 7. Reproducibility Rules
- Keep model, split, and preprocessing versions immutable per benchmark run.
- Do not compare runs across different split versions as a strict regression check.
- Record date/time and source data manifest.

## 8. Initial Coverage Plan
- Phase 1 (fully specified): Sogdian printed + handwritten.
- Phase 2: Old Turkish alignment to same benchmark policy.
- Phase 3: Greek/Latin/Syriac/Coptic profile onboarding.
