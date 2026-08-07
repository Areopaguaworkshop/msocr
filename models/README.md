# Model Storage

Place local Kraken HTR model files in this folder. Kraken 7 training writes `.safetensors`; older compatible runtime models may still use `.mlmodel`.

Current convention:

- Default Sogdian manuscript HTR model:
  `models/kraken/sogdian_manuscript.mlmodel`
- Current Christian Sogdian fine-tune convention:
  `models/kraken/c2av_finetune.safetensors`
- Runtime overrides can point at any compatible Kraken model via:
  - `MSOCR_HTR_RUNTIME_MODEL_PATH`
  - `MSOCR_HTR_MODEL_PATH`
  - `MSOCR_RUNTIME_MODEL_PATH`
  - CLI/API `--model` arguments where supported

`msocr train-remote --output-model` should use a `.safetensors` path. Runtime model resolution accepts either compatible format.

`models/` is gitignored for large binary artifacts. Keep only small placeholders and documentation in git.
