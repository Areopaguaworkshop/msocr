# Training and Benchmark Code Review Template (`code_review.md`)

## 1. Review Scope
Use this checklist for any change that affects:
- training configuration
- preprocessing behavior
- inference settings
- evaluation/benchmark logic
- data split definitions

## 2. Blocking Findings (Must Fix)
- Data leakage across train/val/test (especially manuscript overlap).
- Route confusion between `printed` and `handwritten` workflows.
- Metric computation inconsistency (CER/WER tokenization mismatch without documentation).
- Unversioned benchmark artifacts or missing split/model identifiers.
- Silent fallback behavior that does not set `needs_manual_review` when required.

## 3. Functional Correctness Checklist
- Is `writing_mode` explicit and respected end-to-end?
- Is language direction (`LTR`/`RTL`) applied consistently?
- Are confidence thresholds and fallback conditions explicit?
- Are planned/external modules clearly marked and not treated as implemented?

## 4. Evaluation Correctness Checklist
- Are both `CER` and `WER` produced?
- Is `CER` used as the primary pass metric?
- Are acceptance thresholds aligned with route policy?
- Are benchmark comparisons done on the same split version?

## 5. Reproducibility Checklist
- Split manifests versioned and immutable?
- Model/config/preprocessing versions recorded?
- Randomness controls documented (if applicable)?
- Benchmark metadata complete and queryable?

## 6. Integration Checklist for Current `msocr`
- Compatible with CLI flow: `preprocess -> annotation -> train -> ocr`.
- Mapping to current modules remains valid:
  - `msocr/cli.py`
  - `msocr/data/manager.py`
  - `msocr/data/annotation.py`
  - `msocr/preprocessing/pipeline.py`
  - `msocr/models/inference.py`
  - `msocr/training/ketos_trainer.py`

## 7. Review Output Format
For each review, output:
- `severity`: critical/high/medium/low
- `location`: file path and line (if available)
- `issue`: concise problem statement
- `impact`: benchmark or production risk
- `required_action`: concrete fix request

## 8. Residual Risk Section
If no blocking findings are present, still report:
- remaining risks
- testing gaps
- assumptions requiring future validation
