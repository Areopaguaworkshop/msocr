# HTR Fine-Tuning Data Practices — Research Notes

Compiled 2026-07-06 from @librarian research (ses_0c6ae5368ffeD5OgTHEXvltK1p).
For the msocr Sogdian manuscript HTR project: 178 annotated lines total,
fine-tuning Kraken from a Syriac base model.

## TL;DR for our project

Our splits (154 train / 5 val / 19 holdout) are **below every recommended
minimum**. The 5-line validation set is pure noise for early stopping. The
19-line holdout gives CER confidence intervals of ±4–7 pts, meaning the
71.43% → 80.09% regression we saw in §1 is real in direction but noisy in
magnitude. **Recommended fix: k-fold cross-validation** (no new annotation
needed) — merge val + holdout (24 lines) into one pool, run 5-fold CV
(~142 train / 36 val per fold), report mean CER ± std.

## 1. Train/val/test split ratios

| Source | Train | Val | Test |
|---|---|---|---|
| **Kraken docs (all 4.x–7.x)** | 80% | 10% | 10% |
| Barrere et al. 2024 (Transformer HTR) | 90% | 10% | — (k-fold CV) |
| Tarride et al. 2023 (crowdsourced HTR) | 78.9% | 9.4% | 11.7% |
| IAM / READ2016 / LAM benchmarks | 75–80% | 10–12% | 10–15% |

Kraken default `--partition 0.9` = 90% train / 10% val, with separate
`ketos test` on a held-out 10%. When you pass explicit `-e/--evaluation-files`,
partition is set to 1.0 (no auto-split).

**Range: 80/10/10 to 90/10/0 (k-fold CV replaces fixed test).**

## 2. Minimum training line count

### Fine-tuning (transfer learning)

- **Pippi et al. 2023** (arXiv:2305.02593): fine-tuning works with **as few
  as 5 lines** for single-writer adaptation. Below 230 lines, from-scratch
  training did not converge — pretraining is required.
- **Kohút & Hradiš 2023** (arXiv:2302.06308): fine-tuning "surprisingly
  resistant to overfitting even for an extremely low number of text lines...
  even a single adaptation text line without augmentations improved
  transcription accuracy by 5% on average."
- **Aradillas et al. 2018**: 350 lines with transfer learning → CER 3.3%
  vs 18.2% from scratch (Washington dataset).
- **Riksarkivet/satrn_htr**: "50–60 transcribed pages is enough to halve
  the CER... transcribe 50–100 pages, and finetune."

### From scratch

- Kraken docs: ~50 epochs for "reasonably sized datasets"; tutorial example
  uses 788 train + 88 val lines.
- Reul et al. 2018: 10K lines can match 20K-line performance with good
  architecture + augmentation.
- Transkribus community: 25–75 pages for printed text, more for handwritten.

### Summary table

| Scenario | Min lines | Recommended | Source |
|---|---|---|---|
| Fine-tune (same script, new hand) | 5–15 | 50–200 | Pippi 2023, Kohút 2023 |
| Fine-tune (related script, e.g. Syriac→Sogdian) | 50–100 | 150–500 | Extrapolated |
| Fine-tune (new script, new hand) | 100–200 | 300–1000 | Aradillas 2018, Barrere 2024 |
| From scratch | 1,000+ | 5,000–10,000+ | Reul 2018, Kraken docs |

**Our 154 train lines for Syriac→Sogdian is in the viable-but-tight range.**
The issue is not the training size — it's the val/test splits.

## 3. Validation set size — is 5 lines too few?

**Yes, unequivocally.**

1. Kraken docs warn: small val sets miss characters present in training,
   causing misleading CER signals. "Increasing the size of the validation
   set will often remedy this warning."
2. With 5 lines, a single misrecognized character swings CER by several
   percentage points. Early stopping picks essentially random checkpoints.
3. Kohút & Hradiš 2023 used cross-validation for small adaptation sets
   (1–256 lines) precisely because single small val sets are unreliable.
4. Barrere et al. 2024: for 20–70 line pages, used **4-fold CV** rather
   than a fixed val split.

### Recommended minimums

| Source | Min val | Context |
|---|---|---|
| Kraken docs example | 88 lines | 788-train example |
| IAM benchmark | 976 lines | 6,482 train |
| READ2016 benchmark | 1,040 lines | 8,349 train |
| Barrere 2024 | 10% of data, or k-fold CV | Small datasets |
| **Practical rule of thumb** | **≥50 lines or ≥500 characters** | Stable CER signal |

**Our 5-line val (~2.8% of data) is far below any threshold. CER signal is
dominated by noise.**

## 4. Test/holdout set size — is 19 lines enough?

**No — 19 lines is too few for a reliable CER estimate.**

### Guyon et al. 1998 (canonical test-set sizing paper)

Rule of thumb: `n = 100 / p`, where n = test set size, p = error rate.

| CER range | Min test characters | Min test lines (~30 char/line) |
|---|---|---|
| ~10% CER | 1,000 | ~33 |
| ~5% CER | 2,000 | ~67 |
| ~2% CER | 5,000 | ~167 |

At 95% confidence, 1,000 test chars → true error could be up to **1.25×**
observed. A 0.3× difference between two models is needed for significance.

### Our 19 lines ≈ 570 characters

- If observed CER = 10%, 95% CI is roughly **±4 pts** (true CER 6–14%).
- CER differences < 3 pts between models are **not statistically significant**.

So our 71% → 80% jump (9 pts) is **likely real in direction** but the
magnitude has wide error bars. The 71% baseline itself has CI roughly
±5–7 pts.

### Recommended minimums

- For CER 5–15%: **≥30–50 test lines** (~1,000+ chars).
- For tighter CER estimates: more.

## 5. Strategies for small datasets (~180 lines)

Priority order (all from consensus in recent HTR papers):

### 1. k-Fold cross-validation (strongest recommendation)

- Barrere 2024: 4-fold CV for 20–70 line pages.
- Kohút & Hradiš 2023: CV for 1–256 adaptation lines.
- Aradillas 2020: k-fold CV to detect and purge mislabeled lines.

**For our 178 lines: 5-fold CV → ~142 train / 36 val per fold.** Train 5
models, average holdout CERs. Uses all data for both training and eval.

### 2. Leave-one-out CV

Nearly unbiased but expensive (178 runs). Only worth it for the most
precise CER estimate.

### 3. Data augmentation

Kraken `--augment` (albumentations). Kohút & Hradiš 2023: augmentation
(geometry + blur + noise masking) gave **1.5× larger improvement** over
fine-tuning without. **Aradillas 2021 caution**: apply augmentation to the
**source** pretraining, not the small target set — can hurt on target.

### 4. Synthetic data generation

Pippi 2023: styled HTG (Handwritten Text Generation) models matching target
handwriting style. Requires 15 word images as style examples. Heavy infra
but can dramatically boost results.

### 5. Fixed-iteration stopping (no val set needed)

Kohút & Hradiš 2023: estimate fixed ratio of fine-tuning iterations to
adaptation lines, **outperforming cross-validation** for stopping:

> "fine-tuning for new documents can be performed with a predefined number
> of iterations conditioned only on the amount of available target data."

- 16 lines: ~30 iterations/line
- 256 lines: ~9 iterations/line
- **Our 154 lines: ~1,500–2,000 iterations** as a stopping point.

### What NOT to do

- Don't use a 5-line val set for early stopping — signal is pure noise.
- Don't trust a 19-line test set for final CER — CI too wide.
- Don't apply heavy augmentation to the small target set (Aradillas 2021).

## 6. Kraken/ketos version-specific guidance (7.x)

### Recommended fine-tune recipe

```
ketos train --resize new -i base_model.mlmodel \
  --warmup 5000 --freeze-backbone 1000 training_data/*
```

| Parameter | Recommendation | Source |
|---|---|---|
| `--resize new` | **Always `new`, not `union`** — union causes "rapidly unlearn missing labels" | Kraken docs |
| `--warmup` | ≥ a couple epochs (e.g., 5000 steps) | Kraken docs |
| `--freeze-backbone` | 1–2 epochs (e.g., 1000 steps) when base dissimilar/pretrained | Kraken docs |
| `--augment` | Requires `albumentations`. Cautious on small target sets | Kraken + Aradillas 2021 |
| `--schedule cosine` | "Start by finetuning from the default model for a fixed number of epochs (50 for reasonably sized datasets) with a cosine schedule" | Kraken docs |
| `--quit early` | Default. Early stopping on val CER | Kraken docs |
| `--partition 0.9` | Default 90/10 train/val split | Kraken CLI |

### Critical Kraken warning

> "Fine-tuning models from pre-trained weights is quite a bit less stable
> than training from scratch or fine-tuning an existing model. As such it
> can be necessary to run a couple of trials with different hyperparameters
> (principally learning rate) to find workable ones. It is entirely possible
> that pretrained models do not converge at all even with reasonable
> hyperparameter configurations."

### Slicing (for highly dissimilar alphabets)

If Syriac→Sogdian alphabet mismatch is large, Kraken supports slicing off
and reinitializing the last layers:

```
ketos train -i base_model.mlmodel --append 7 -s '[Cr3,3,64 Do0.1]' training_data/*
```

More aggressive than `--resize new`. May be needed if scripts are
substantially different.

## Diagnosis of our current setup

**Splits: 154 train / 5 val / 19 holdout (178 total)**

The §1 freeze-rows regression (80% vs 71% baseline) is likely caused by:

1. **5-line val set is noise.** Early stopping picked a checkpoint based on
   random fluctuation. The "best" model may actually be worse than baseline.
2. **Overfitting to 154 lines.** With tiny val, no reliable signal to stop
   before memorization.
3. **`--resize union` may be wrong.** Kraken docs say use `new` not `union`
   — union preserves Syriac chars the model should forget. This matches our
   §1 failure mode (shared-class errors).

## Recommended fix (immediate, no new annotation)

1. **Merge val + holdout into one pool** (24 lines: 5 + 19).
2. **5-fold cross-validation:** ~142 train / 36 val per fold.
3. **Use `--resize new`** (not `union`).
4. **Use `--warmup 5000 --freeze-backbone 1000`** (1–2 epochs freeze).
5. **Report mean CER ± std across folds** as the final metric.
6. **Try fixed-iteration stopping** (~1,500–2,000 iterations) as an
   alternative to early stopping.

## If we can annotate more

- Add 30–50 more lines for a proper holdout of 50+ lines.
- Target: ~150 train / 30 val / 50 test — tight but workable.
- Guyon 1998 rule: for CER ~10%, need ~1,000 test chars ≈ 33 lines.

## Citations

- Kraken docs: https://kraken.re/5.3.0/ketos.html, https://kraken.re/6.0.0/tutorials/training.html
- Kraken GitHub: https://github.com/mittagessen/kraken
- Pippi et al. 2023: https://arxiv.org/abs/2305.02593
- Kohút & Hradiš 2023: https://arxiv.org/abs/2302.06308
- Aradillas et al. 2018, 2020, 2021 (multiple papers)
- Barrere et al. 2024 (Transformer HTR)
- Tarride et al. 2023 (crowdsourced HTR)
- Reul et al. 2018
- Hodel et al. 2021
- Guyon et al. 1998: https://people.sabanciuniv.edu/berrin/cs512/reading/guyon-datasize.pdf
- Riksarkivet/satrn_htr: https://huggingface.co/Riksarkivet/satrn_htr
- HTR-United: https://htr-united.github.io/

## Files this research informs

- `docs/fixa-v4-result-2026-07-06.md` — §1 failure report (confirms the
  5-line val set is a confounding factor)
- `docs/fixa-v4-plan-2026-07-06.md` — original v4 plan (§1 falsified; §3/§4
  now primary)
- `data/manifests/c2av-finetune.json` — current split (154/5/19, needs
  redesign per §4 LOOCV or 5-fold CV)
- `msocr/training/ketos_trainer_api.py` — freeze harness (keep for
  ablations but not the default)
- This file: `docs/htr-finetuning-data-practices-2026-07-06.md`