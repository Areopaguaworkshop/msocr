# Model Storage

Place local Kraken HTR model files in this folder.

Current convention:

- Default Sogdian manuscript HTR model:
  `models/kraken/sogdian_manuscript.mlmodel`
- Runtime overrides can point at any compatible Kraken `.mlmodel` via:
  - `MSOCR_HTR_RUNTIME_MODEL_PATH`
  - `MSOCR_HTR_MODEL_PATH`
  - `MSOCR_RUNTIME_MODEL_PATH`
  - CLI/API `--model` arguments where supported

`models/` is gitignored for large binary artifacts. Keep only small placeholders and documentation in git.
