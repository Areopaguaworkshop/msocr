# Strat-here: msocr Instruction Baseline

## Purpose
- This folder defines the general documentation-first agent specification for `msocr`.
- You should only write agent mode Markdown and skills, pipeline, source_registry in yaml only, do not write any code.

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
- Do not write code, only do pan with Markdown and Yaml files. 
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

## Next Step
- Design and maintain the blueprint in Markdown and YAML only.
- Execute language and mode expansion in this order:
  1. Greek and Latin printed OCR
  2. Syriac and Coptic printed OCR
  3. Greek and Latin handwritten HTR
  4. Syriac and Coptic handwritten HTR
  5. Armenia and Geez (both printed OCR and handwritten HTR)
  6. Sogdian and Old Turkish, with primary focus on handwritten HTR
- Keep printed OCR and handwritten HTR routes separate in all skills and pipelines.
