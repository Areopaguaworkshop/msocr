# Kraken HTR Fine-Tuning Feasibility: 178 Lines on 12 c2av Plates

**Date:** 2026-07-04
**Question:** Is 178 transcribed lines across 12 c2av plates enough to fine-tune a Kraken HTR model from the Sophro Mhiro Syriac base?

## Answer

**Yes, barely — fine-tuning only.** From-scratch training will not converge at this scale. Fine-tuning from the Sophro Syriac base is viable because the visual backbone already reads Syriac letterforms; only the output head is rebuilt for the 24-character Latin-transliteration codec.

| | From scratch | Fine-tune from sophro |
|---|---|---|
| Minimum lines | 800–2,000+ | ~71–700 (178 is in range) |
| Expected CER | will not converge | **20–40%** |

## Data inventory

| Manuscript | Plates | Lines | Script |
|---|---|---|---|
| c2av plates 1–12 | 12 | 178 | Latin transliteration (24 chars) |
| e57d259b frag_004 | 1 | 18 | Latin transliteration |
| c2av plate 19 | 1 | 1 (of 11) | Syriac (`ܫܪܥܦ`) |
| **Total annotated** | | **197** | |

c2av char inventory (24 chars): `. b c d f g l m n p q r s t w x y z š ž ʾ γ θ` — none overlap the Sophro 36-char Syriac codec (ܐ–ܬ + punctuation/diacritics).

## Minimum line counts (from Kraken docs, issues, papers)

### From-scratch
- **Kraken docs v6.0**: ~800 lines for small-grapheme printed scripts; manuscripts need more.
- **Kraken Issue #711** (maintainer): "a couple of thousand well-sampled lines" for regular script.
- **Pippi et al. 2023** (arXiv:2305.02593): no convergence below ~230 lines from scratch.
- **Skidzun/kraken_training**: ~10,000 words per scribal hand for a base model.

### Fine-tuning
- **Kraken Issue #711**: 700 lines fine-tuned from CATMuS → ~86–90% character accuracy in <50 epochs.
- **La Paperista blog (2026)**: 71 lines fine-tuned from CATMuS → usable model in 20 epochs.
- **OICEN-HTR (Old Norse)**: hundreds of lines from CATMuS-medieval → 95–99% CA.

## Sophro Mhiro base model statistics

From Zenodo record (DOI 10.5281/zenodo.17406773):

| Metric | Value |
|---|---|
| Training lines | 3,466 |
| Training characters | 90,991 |
| Manuscripts | 38 (6th–20th c.) |
| Scripts | Serto, Estrangela, Eastern Syriac |
| Test accuracy | 97.4% character accuracy |
| Codec | 36 chars: Syriac consonants + punctuation/diacritics, no vowels |

Independent evaluation (Benabdellah 2026, ETCBC): 28.10% CER on out-of-domain Syriac with automatic segmentation; ~6% CER with manually corrected segmentation.

The base was trained on **19.5× more lines** and **34× more characters** than our dataset, across **38 manuscripts** vs. our likely 1.

## Binding constraint: single-manuscript diversity, not line count

12 plates from one manuscript (c2av) = one scribal hand, one page layout, one imaging condition. The model will overfit to that hand. Kraken maintainer (Issue #711):

> "It is better to create a dataset of 100 pages from 1 page taken from 100 different documents each than by transcribing 100 pages from a single one."

Sophro used 38 manuscripts for 3,466 lines; we have 1 manuscript for 178 lines.

**Mitigation:** augmentation (built-in `albumentations`), `--freeze-backbone`, early stopping. Cannot substitute for diversity, but helps.

## What 178 lines gives you

- **Usable first-pass model** for accelerating manual transcription of the remaining ~87 c2av plates. Correcting 60–80% accurate output is faster than typing from scratch.
- **NOT a publishable model.** Publishable HTR reports CER < 5–10%. We would need 500–1,000+ lines across multiple manuscripts.

## Recommended fine-tune configuration

```
ketos train --load sophro_mhiro_syriac.safetensors --resize new \
  --freeze-backbone 5000 --warmup 200 --augment \
  --min-epochs 20 --lag 10 -r 1e-4
```

- `--resize new`: rebuild output head for 24-char Latin codec.
- `--freeze-backbone 5000`: keep visual CNN frozen for first 5,000 steps; only train new head.
- `--warmup 200`: gradual LR ramp (critical for fine-tuning stability).
- `--augment`: built-in `albumentations` pipeline (PixelDropout, MotionBlur, MedianBlur, Blur, OpticalDistortion, ElasticTransform, SafeRotate ±3°). 50% augmentation probability per image. Single biggest overfitting mitigation at this scale.
- `-r 1e-4`: lower than default 1e-3 (recommended for small datasets).

## Split recommendation

At 178 lines the standard 80/10/10 gives 142/18/18 — too few val lines for reliable early stopping.

**Chosen: 10 train / 1 val / 1 holdout** (~148/15/15 lines):
- Train: plates p-01 … p-10
- Validation: plate p-11 (early stopping signal, noisy but honest)
- Holdout: plate p-12 (one real generalization number on an unseen plate)

Alternative (not chosen): leave-one-plate-out CV, 12× the cost, more principled but expensive.

## Augmentation details (kraken DefaultAugmenter)

| Transform | Prob | Purpose |
|---|---|---|
| PixelDropout | 20% | ink degradation, noise |
| MotionBlur | 20% | camera motion / page movement |
| MedianBlur | 10% | ink bleed |
| Blur | 10% | out-of-focus |
| OpticalDistortion | 30% | page curvature / lens distortion |
| ElasticTransform | 10% | parchment warping |
| SafeRotate (±3°) | 20% | slight page rotation |

Overall augmentation probability: 50% per image.

## Sources

- Kraken training docs: https://kraken.re/6.0.0/tutorials/training.html
- Kraken recognition training: https://kraken.re/6.0.0/training/rectrain.html
- Kraken Issue #711: https://github.com/mittagessen/kraken/issues/711
- Skidzun training guide: https://github.com/Skidzun/kraken_training
- Pippi et al. 2023: https://doi.org/10.48550/arxiv.2305.02593
- La Paperista: https://aishapectyo.com/2026/05/02/first-pass-handwriting-ocr-on-1900s-manuscripts/
- Sophro Mhiro Zenodo: https://doi.org/10.5281/zenodo.17406773
- ETCBC assessment: https://etcbc.nl/methodology/htr-for-syriac-manuscripts-an-assessment-as-of-march-2026/

## Time estimate (RunPod RTX 3090)

| Phase | Smoke (1 plate, 1 epoch) | Real (10 plates, 30 epochs) |
|---|---|---|
| Pod boot | ~3 min | ~3 min |
| pip install kraken | ~5 min | ~5 min |
| Setup patches | ~1 min | ~1 min |
| Upload data | ~1 min (1 img + 1 xml) | ~3 min (10 imgs + 10 xmls) |
| Training | ~1 min (1 epoch × 15 lines) | ~15–30 min (30 epochs × 148 lines + augment) |
| Download artifact | ~1 min | ~1 min |
| Pod terminate | ~1 min | ~1 min |
| **Total** | **~12–15 min** | **~30–45 min** |

Based on observed pod behavior in the e57d259b smoke test (pod boot ~3 min, pip install ~5 min, 2 epochs on 18 lines < 1 min).