# Kraken HTR Training Techniques — Research Synthesis

> Deep research on Kraken 7.0.x training parameters and fragmented-manuscript HTR
> state-of-the-art, synthesized to guide Christian Sogdian (C2AV) model training
> and the broader msocr Sogdian pipeline.
>
> Companion to:
> - [`kraken-training-data-research.md`](./kraken-training-data-research.md) —
>   dataset-size requirements (lines → CER tiers)
> - [`kraken-fragmentary-manuscripts.md`](./kraken-fragmentary-manuscripts.md) —
>   engineering reference for fragmentary folios (BLLA failure modes, split-baseline
>   annotation convention, lacunae handling)
> - [`multi-script-htr-research.md`](./multi-script-htr-research.md) — why the
>   three Sogdian scripts need separate models
> - [`runpod-gpu-research.md`](./runpod-gpu-research.md) — GPU selection and cost
>
> Sources: @librarian web research (Kraken 7.0.x docs, GitHub issues, published
> papers, 2023–2026) + @explorer summary of the in-repo docs above. June 2026.

## TL;DR

- **Architecture**: The Kraken 7.0 default recognition spec
  `[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]`
  (4 conv blocks → 3 bidirectional LSTM @ 200) is the sweet spot for manuscripts.
  Keep it for Sogdian. Deviate only with >10k lines or unusual glyph inventory.
- **Fine-tuning, not from-scratch, for Christian Sogdian**: Fine-tune from
  `sophro_mhiro_syriac.mlmodel` (Syriac base) with `--resize new`,
  `--base-dir R`, `--normalization NFD`, `--schedule cosine --warmup 200`,
  `--min-epochs 50 --lag 15 --quit early`, `--augment`, `--precision bf16-mixed`.
- **Current `sogdian_config.yaml` has 4 flags that fight the research**:
  `normalization: NFC` (should be NFD for diacritic-heavy RTL), `weight_decay: 0.0001`
  (should be 0 for CTC recognition), `schedule: 1cycle` (should be cosine for
  fine-tuning), `binarize: true / otsu` (should be off — Kraken reads grayscale,
  binarization destroys faint ink). See §11.
- **Fragmented manuscripts**: CTC has no lacuna token. TrOCR research shows a
  lacuna token + synthetic gaps restores 65% of lacunae, but this is not
  portable to Kraken. Best current Kraken path: split-baseline annotation
  (one baseline per inked fragment), fine-tune seg model on 5–20 fragmentary
  pages, keep diplomatic GT. Track Orli (Bézier-curve line predictor, 2026) as
  the upcoming BLLA replacement that may handle gaps better.
- **Self-supervised pretraining** (`ketos pretrain`, Vogler et al. 2022) helps
  recognition with as few as 30 labeled lines but is unstable and does not
  help segmentation. Reserve for when labeled data is extremely scarce.

---

## 1. VGSL Architecture

### 1.1 Default 7.0 recognition spec

```
[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]
```

- Input: `[batch=1, height=120, width=variable, channels=1(grayscale)]`
- 4 conv blocks: 32 → 32 → 64 → 64 channels, kernels 13 → 13 → 9 → 9
- 2D dropout (0.1) after each conv block
- `S1(1x0)1,3`: reshape collapses height into channels before RNN
- 3 bidirectional LSTM layers (200 units each), 2D dropout (0.1) between
- ~4.1M params, ~16MB on disk

Source: `kraken/lib/default_specs.py` ([GitHub](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/default_specs.py)), [VGSL docs](https://kraken.re/5.1/vgsl.html).

### 1.2 When to deviate

| Situation | Change | Rationale |
|-----------|--------|-----------|
| Small dataset (<1000 lines) | Reduce LSTM to 100–128 units, conv channels to 16→32→48→64 | Prevents overfitting |
| >10k lines, complex script | Add a 5th conv block | Hierarchical features need depth |
| Heavy diacritics (Syriac, Arabic, Sogdian) | Keep 3 LSTM layers; do NOT reduce channels below 32 | Diacritics are small features — need CNN depth |
| RTL connected script | `--base-dir R`; wider conv kernels (13→15) help | Wider receptive field for connected forms |
| Tall ascenders/descenders | Increase input height 120 → 180–200 | Default 120 may clip Syriac/Arabic ascenders |

**For Christian Sogdian (Syriac script, C2AV plates):** keep the default spec.
Syriac has diacritics but not extreme ascender height; 120px input is adequate.
The default is what the `sophro_mhiro_syriac` base model uses, so keeping it
avoids architecture mismatch on fine-tune.

### 1.3 Segmentation spec (BLLA)

```
[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]
```

- Input: 1800px height, RGB
- GroupNorm (`Gn32`) instead of Dropout
- Strided convs (stride 2,2) instead of MaxPool
- Bidirectional LSTMs in both x and y
- ~2–3× more VRAM than recognition spec

Source: `kraken/lib/default_specs.py`.

---

## 2. Optimizer & Schedule

### 2.1 Defaults vs recommendations

| Parameter | Rec default | Seg default | Recommendation for Sogdian fine-tune |
|-----------|------------|-------------|--------------------------------------|
| Optimizer | Adam | Adam | Adam (AdamW not needed — weight decay 0 for rec) |
| Learning rate | 1e-3 | 2e-4 | **1e-4** (fine-tune); 1e-3 for from-scratch |
| Weight decay | 0 | 1e-5 | **0** for recognition; 1e-5 only for seg |
| Schedule | constant | constant | **cosine** for fine-tune; `1cycle` for from-scratch |
| Warmup | 0 | 0 | **200** steps (fine-tune similar domain); 5000 (pretrain→finetune) |
| Gradient clipping | off | off | Off (Kraken doesn't expose by default) |

Sources: `default_specs.py`, [kraken.re/6.0.0/training/rectrain.html](https://kraken.re/6.0.0/training/rectrain.html), [Digital Orientalist](https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/).

### 2.2 Schedule selection

| Schedule | When | Notes |
|----------|------|-------|
| `constant` | Default, safe | Boring but works |
| `1cycle` | **From-scratch** | Cycle length = `--epochs`; faster convergence on small data |
| `cosine` | **Fine-tuning** | Use with `--cos-t-max` (epochs) and `--cos-min-lr` (1e-7); stabilizes fine-tune |
| `reduceonplateau` | Conservative | `--rop-factor 0.1 --rop-patience 5`; safe fallback |
| `exponential`, `step` | Rarely used | — |

**Christian Sogdian is fine-tuning from a Syriac base → cosine + warmup 200.**

---

## 3. Batch Size, Epochs, Early Stopping

### 3.1 By dataset size

| Dataset size | Batch | Min-epochs | Lag | Quit | Notes |
|-------------|-------|-----------|-----|------|-------|
| 1–50 pages (~200–2000 lines) | 4–8 | 50 | 15 | `early` | Fine-tune only; `--resize new` |
| 50–500 pages (~2000–20k lines) | 8–16 | 20 | 10 | `early` | Fine-tune or from-scratch for simple scripts |
| 500+ pages (~20k+ lines) | 16–32 | 10 | 10 | `early` | From-scratch viable; use binary format |

### 3.2 `--quit` modes

- **`early`** (rec default): stop when val error stops improving for `--lag` epochs. Use for all fine-tuning.
- **`fixed`** (seg default): run exactly `--epochs`. Use when you know the right count.
- ~~`dolphin`~~: not a real Kraken option.

### 3.3 `--freeze-backbone` (iterations, not epochs)

| Scenario | `--freeze-backbone` | `--warmup` |
|----------|--------------------|-----------:|
| Fine-tune, similar domain | 0 | 0–200 |
| Fine-tune, dissimilar domain | 1000–5000 | 200–1000 |
| Pretrain → fine-tune | 1000–5000 | 5000 |

**Warning (TrOCR ablation, 2026):** encoder freezing is brittle — freezing past
layer 3 of the encoder becomes harmful. Decoder freezing is safer (up to layer 6).
**For Kraken CRNN: prefer full fine-tuning; only freeze if convergence fails.**

Source: [arXiv:2606.24302](https://arxiv.org/html/2606.24302).

### 3.4 Kiessling's data point (GitHub #711)

A 1000-line mixed-style dataset:
- From scratch, default spec, ~100 epochs → **74% CA** (26% CER)
- Fine-tune CATMuS Medieval 1.6, <50 epochs → **~90% CA** (10% CER)

Takeaway: fine-tuning cuts data needs by ~50% and converges faster.

---

## 4. Augmentation

### 4.1 What Kraken's `--augment` actually applies

`DefaultAugmenter` in `kraken/lib/dataset/recognition.py` (with `p=0.5` overall):

```
ToFloat
PixelDropout(p=0.2)                              # ink degradation, speckling
OneOf [MotionBlur(p=0.2), MedianBlur(3, p=0.1), Blur(3, p=0.1)]  # focus/bleed
OneOf [OpticalDistortion(p=0.3), ElasticTransform(alpha=7, sigma=25, p=0.1),
       SafeRotate(±3°, p=0.2)]                   # warp, skew
```

**Removed in PR #673**: `Affine` translation and `ShiftScaleRotate` (cut borders,
lost ink). Replaced with `SafeRotate`.

Source: [kraken/lib/dataset/recognition.py](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/recognition.py), [PR #673](https://github.com/mittagessen/kraken/pull/673).

### 4.2 What helps for damaged manuscripts

| Transform | Effect | Verdict |
|-----------|--------|---------|
| PixelDropout (p=0.2) | Simulates ink loss, speckling | **Essential** for damaged ink; consider p=0.3 for severe cases |
| ElasticTransform (alpha=7) | Parchment warp, hand variation | **Most effective** single augmentation (TrOCR ablation: CER 1.86) |
| Blur variants | Ink bleed, focus issues | Keep |
| SafeRotate (±3°) | Skewed lines | Keep |
| OpticalDistortion | Camera/lens artifacts | Keep |

### 4.3 What is NOT in Kraken's default `--augment`

These require pre-processing line images before training:

| Transform | Simulates | Evidence |
|-----------|-----------|----------|
| **Erosion** (morphological, kernel 2–3) | Faded ink, thin strokes | TrOCR ablation 2026 |
| **Synthetic lacuna injection** (white rectangles, 5–15% width) | Holes, missing fragments | Borkar & Smith 2024 (TrOCR only — 65% lacuna restoration) |
| **Ink diffusion/bleeding** | Palimpsest degradation | MDPI Mathematics 2025 |
| **Random brightness/contrast (±10%)** | Uneven lighting | Standard |
| **CLAHE** | Contrast normalization | **NOT beneficial** (TrOCR ablation: statistically insignificant) |

**For Sogdian C2AV**: Kraken's default `--augment` covers the essentials. Add
erosion as a pre-processing step if faded ink is a major problem on specific
plates. Do NOT add CLAHE (the current config has it on — see §11).

Sources: [arXiv:2606.24302](https://arxiv.org/html/2606.24302) (TrOCR ablation),
[arXiv:2508.11499](https://arxiv.org/abs/2508.11499) (TrOCR ensemble),
[arXiv:2407.00250](https://arxiv.org/abs/2407.00250) (Mind the Gap).

---

## 5. Mixed Precision & VRAM

### 5.1 Precision

| Precision | When | VRAM (batch 8, default spec) |
|-----------|------|------------------------------|
| `32-true` (default) | CPU, debugging, MPS | ~6 GB |
| **`bf16-mixed`** | **Ampere+ (RTX 30xx+, A100)** | ~3.5 GB |
| `16-mixed` | Older (V100, T4, RTX 20xx) | ~3.5 GB (risk of gradient underflow) |
| `16-true` | Not recommended | Highest instability |

**Use `bf16-mixed` on RTX 3090/4090.** Avoids FP16 underflow, same memory savings.

### 5.2 VRAM by batch size (default rec spec)

| Batch | FP32 | BF16-mixed |
|-------|------|-----------|
| 1 | ~2.5 GB | ~1.5 GB |
| 8 | ~6 GB | ~3.5 GB |
| 16 | ~10 GB | ~5.5 GB |
| 32 | ~18 GB | ~10 GB |

Seg spec uses ~2–3× more (1800px RGB input).

---

## 6. Transfer Learning & Base Model Selection

### 6.1 Base models by target script

| Base model | Domain | Good for fine-tuning to |
|-----------|--------|------------------------|
| **CATMuS Medieval 1.6.0** | European MSS 8th–16th c. | Latin, Greek, Cyrillic |
| **CATMuS Print (Large)** | European prints 16th–21st c. | Printed Latin |
| **openiti-arabic-base** | Printed Arabic-script | Arabic, Persian, Urdu, Ottoman; **also Syriac** (shared script family) |
| **sophro_mhiro_syriac** | Syriac manuscripts | **Christian Sogdian** (Syriac script U+0710) — primary choice |
| **Avestan (Nikyek)** | Avestan MSS | Sogdian national, Pahlavi |
| **blla.mlmodel** (built-in) | General baseline seg | Any script's segmentation |

### 6.2 Cross-script transfer reality

| Source → Target | Feasibility |
|----------------|-------------|
| Arabic → Syriac | **Good** — shared script family, connected cursive |
| Avestan → Sogdian | **Promising** — both Iranian, RTL |
| CATMuS Medieval → Arabic/Syriac/Sogdian | **Poor** — disjoint script, direction, glyphs |
| Syriac → Christian Sogdian | **Excellent** — Christian Sogdian *is* Syriac script + a few added letters |

**Christian Sogdian uses the Syriac alphabet (U+0710 block).** Fine-tuning from
`sophro_mhiro_syriac.mlmodel` is the correct path. CATMuS base would be useless.

### 6.3 `--resize` modes

| Mode | Behavior | When |
|------|----------|------|
| **`new`** | Removes unused chars from base, adds new ones | **Recommended** for fine-tune to different alphabet |
| `union` | Keeps all base chars + adds new | Only when base alphabet must be preserved |
| `both` | Same as `new` | Legacy alias |
| `fail` | Error if alphabets differ | Continuing same-data training |

### 6.4 Minimum data per base

| Base | Min lines for fine-tune | Recommended |
|------|------------------------|-------------|
| CATMuS Medieval | 200–500 | 1000+ |
| openiti-arabic | 500–800 | 2000+ |
| sophro_mhiro_syriac | 200–500 | 1000+ |
| Pretrained (unsupervised) | 30–90 | 500+ |

### 6.5 Pretraining result (Kiessling, HAL 2024)

Unsupervised pretrain (750k unlabeled lines) + N labeled lines on Arabic MSS:
- 0 pretrain lines: 51.0% CER
- 30 labeled: **14.8% CER**
- 90 labeled: **10.0% CER**

Source: [hal-05438436](https://hal.science/hal-05438436v1/file/kraken_ben.pdf).

---

## 7. CTC-Specific Tuning

### 7.1 Unicode normalization

| Option | Effect | Use when |
|--------|--------|----------|
| **`--normalization NFD`** | Decomposes precomposed (é → e + ◌́) | **Diacritic-heavy RTL** (Arabic, Syriac, Sogdian, polytonic Greek) — each diacritic becomes a separate CTC label |
| `NFC` | Composes base+diacritic | Diacritics integral (German umlauts) |
| `NFKD` | Compatibility decomposition | CATMuS models use this |
| `NFKC` | Compatibility composition | Rarely used |
| none | No normalization | GT already consistent |

**Christian Sogdian: NFD.** Syriac has vocalization and diacritic marks
(seyame, vowels, shewa) that should be separate CTC labels for best accuracy.
The current config has `NFC` — this is wrong for the script. See §11.

### 7.2 Reading direction

| Setting | When |
|---------|------|
| `--base-dir L` | LTR (Latin, Greek, Cyrillic) |
| **`--base-dir R`** | **RTL (Arabic, Syriac, Hebrew, Sogdian, Avestan)** |
| `auto` (default) | Auto-detect from XML metadata — **fails if XML lacks direction metadata** |

**For Sogdian: always set `--base-dir R` explicitly.** Wrong base direction →
garbled output (BiDi resolves logical→display order wrong).

### 7.3 Line height & pad

- **Input height**: 120px in default spec. `S1(1x0)1,3` collapses full height
  into channels. Keep 120 for Sogdian; use 180–200 for tall-ascender scripts.
- **`--pad 16`** (default): 16px white padding L/R of each line. Prevents CTC
  forcing a character at image edge. Increase to 32 for wide initial/final
  forms (Arabic, Syriac connected forms).

---

## 8. Recognition vs Segmentation Training

### 8.1 When to train what

| Scenario | Train seg? | Train rec? |
|----------|:----------:|:----------:|
| New script, no base model | Yes (scratch) | Yes (scratch/finetune) |
| Base seg works, rec poor | No | Yes (finetune) |
| Base seg misses lines | Yes (finetune) | Maybe |
| New layout (columns, marginalia) | Yes (finetune, class map) | No |

### 8.2 Seg fine-tune parameters

| Parameter | Recommendation |
|-----------|----------------|
| `--epochs` | 50–100 (finetune); 100–200 (scratch) |
| `--quit` | `fixed` for seg |
| `--lrate` | 2e-4 |
| `--suppress-regions` | true if only baseline training |
| `--topline` | false = baseline; true = topline; null = centerline |
| `--resize` | `new` for class changes |

### 8.3 Seg dataset sizes

- Baseline-only fine-tune: **5–10 pages** with corrected baselines
- Region + baseline: **20–50 pages**
- From scratch: **100+ pages** diverse layouts

### 8.4 Polygon offset regression (Kraken #773)

Kraken 7.0.0b6 hardcoded `offset=4` in `calculate_polygonal_environment()`,
replacing K6's model-derived `offset=8`. The narrower polygon clipped character
edges → **2–3% CER regression** on dense MSS. **Fixed in 7.0 stable.**

Takeaway: if CER is unexpectedly high, check seg model's `line_width` matches
what the polygon extractor uses. Kraken 7.0.2 (current) has the fix.

---

## 9. Fragmented Manuscript HTR — State of the Art

This section extends [`kraken-fragmentary-manuscripts.md`](./kraken-fragmentary-manuscripts.md)
with published 2023–2026 research. The engineering reference there remains
canonical for msocr's annotation and segmentation policy.

### 9.1 Lacuna handling — CTC vs transformer

**CTC (Kraken, PyLaia): no lacuna token exists.** The CTC blank is an alignment
mechanism ("no char at this timestep"), not a semantic gap marker. When a line
polygon encloses physically blank (damaged) pixels, CTC collapses them to
nothing — the transcript looks complete but text is silently missing.

**TrOCR + lacuna token (Borkar & Smith, ICDAR 2024 Workshop):**
- Add a dedicated lacuna token to vocabulary
- Train on binarized line images with synthetic white rectangular gaps
- **65.85% lacuna restoration** vs 5.6% without lacuna knowledge
- Tradeoff: clean-char accuracy 77.82% → 75.38%
- Log-probability flagging: detects lacuna-containing lines 53% of the time

**MsBERT (ACL 2024 ML4AL):** `[GAP]` and `[ONEGAP]` tokens for *post-HTR*
language-model gap filling in Hebrew MSS. Not recognition-time.

**Implication for Kraken/Sogdian:** No way to train a lacuna-emitting CTC model.
The split-baseline annotation convention (one baseline per inked fragment) in
`kraken-fragmentary-manuscripts.md` is the correct workaround — damaged regions
never enter a line polygon, so CTC never sees them.

Sources: [arXiv:2407.00250](https://arxiv.org/abs/2407.00250), MsBERT ACL 2024.

### 9.2 Segmentation on damaged pages

**BLLA failure modes** (documented in Kraken issues, see `kraken-fragmentary-manuscripts.md`):
- #745: holes, faded ink, darkened parchment → excessive horizontal splitting
- #677: split lines despite 40-page training set; fixed by fine-tuning from
  `ubma_segmentation.mlmodel` (UB Mannheim) where BLLA fine-tune failed
- #544: bottom-of-region polygons truncated; workaround = dummy-line insertion
- #773: K7.0.0b6 offset=4 regression (fixed in stable)

**Current best practice for fragmentary Sogdian:**
1. Start from `blla.mlmodel` or `ubma_segmentation.mlmodel`
2. Fine-tune on 5–20 corrected fragmentary pages, 40–50 epochs, `--quit fixed`
3. `--suppress-regions` if region data is inconsistent (baseline-only)
4. `--resize new` if adding/changing line types

**No published seg fine-tuning CER numbers for fragmentary corpora** (DSS,
papyri, palimpsests). Kiessling's "fine-tune with a couple of pages" advice is
anecdotal, not benchmarked.

### 9.3 DSS / papyri / palimpsest work (no end-to-end HTR yet)

| Project | What it solves | HTR CER |
|---------|---------------|---------|
| DSS fragment segmentation (Brown-deVost et al. 2024) | Fragment-from-background segmentation, classical CV | — |
| DSS MTEM (Kurar-Barakat & Dershowitz 2024/2026) | Ink vs parchment via multispectral thresholding; ink F1=0.77, parchment F1=0.99 | — |
| Zenon Papyri (2022) | Greek papyri diplomatic GT as PageXML | not published |
| Byzantine Greek HTR (ML4AL 2024) | Swin+BERT, 142M params, 75 epochs | CERr 9.73 (28 pages) → 15.72 (70 pages) |
| Palimpsest GAN (MDPI 2025) | Image reconstruction, not recognition | — |

**Biggest open gap: no one has published a fragment image → text-line seg → HTR → CER pipeline for any fragmentary corpus.**

Sources: [arXiv:2406.15692](https://arxiv.org/abs/2406.15692), [arXiv:2411.10668](https://arxiv.org/abs/2411.10668), MDPI Mathematics 13(14):2304.

### 9.4 Self-supervised pretraining (`ketos pretrain`)

Implements Vogler et al. NAACL 2022 (contrastive masked-patch pretraining):

```bash
ketos pretrain --mask-width 4 --mask-probability 0.2 --num-negatives 3 -f binary unlabeled.arrow
ketos train -i pretrain_best.mlmodel --warmup 5000 --freeze-backbone 1000 --resize new -f binary labelled.arrow
```

- Helps recognition with as few as **30 labeled lines**
- **Unstable**: "entirely possible that pretrained models do not converge at all"
- **Recognition only** — no `ketos segpretrain`
- `--warmup` + `--freeze-backbone` **required** for convergence
- Use `ketos compile --keep-empty-lines` to include untranscribed lines

**For Sogdian: reserve for when labeled data is extremely scarce (<100 lines).
With the Syriac base model available, supervised fine-tuning is more reliable.**

Source: [arXiv:2112.08692](https://arxiv.org/abs/2112.08692), [Kraken pretrain docs](https://kraken.re/6.0.0/training/rectrain.html).

### 9.5 Ensembling

TrOCR augmentation ensemble (2025): top-5 voting of models trained with
different augmentation seeds → CER 1.60 vs best single 1.86 (**14% relative
gain**).

**For Kraken:**
- Worth it with ≥30 pages: train 3–5 variants with different `--augment` seeds,
  majority-vote at character level
- Not worth it below ~30 pages (each model needs enough data to converge)
- Kraken has **no built-in ensembling** — external voting script required
- No published seg-model ensembling for fragments

Source: [arXiv:2508.11499](https://arxiv.org/abs/2508.11499).

### 9.6 Orli — upcoming BLLA replacement

Kiessling, Jun 2026 ([arXiv:2606.04166](https://arxiv.org/abs/2606.04166)):
- ConvNeXtV2-tiny encoder → autoregressive transformer decoder
- Baselines as **cubic Bézier curves** with chord-frame parameterization
- Joint detection + reading order in one pass
- Trained on 196,691 pages, 10 writing systems
- Marginally exceeds cBAD SOTA without dataset-specific training
- Open-source: [github.com/mittagessen/orli](https://github.com/mittagessen/orli),
  weights on Zenodo

**Relevance to fragments:** direct Bezier regression may "jump" across holes
better than pixel-level heatmaps — a continuous curve can span a gap if the
model learns baselines are continuous. **Not yet validated on fragmentary
corpora.** Replaces BLLA in Kraken ≥7.x. Track it.

---

## 10. Known Kraken 7.0.x Bugs & Gotchas

### 10.1 Fixed in 7.0 stable

| Bug | Version | Impact |
|-----|---------|--------|
| Polygon offset=4 regression (#773) | 7.0.0b6 | 2–3% CER increase |
| `blla.segment()` returns micro-regions (#773) | 7.0.0b6 | 483 regions instead of ~40 lines |
| `segtrain` float16 crash (#768) | 7.0b7 | `RuntimeError: dtype float16` |
| Reading-order model load failure | 7.0.0b2 | RO weights fail to load |
| Cosine LR broken in RO training | 7.0.0b2 | LR not applied per-step |

### 10.2 `self.net is None` checkpoint crash

Known issue loading checkpoints from interrupted runs. 7.0 produces `.ckpt`
(Lightning) files; the error occurs when:
1. Checkpoint saved mid-initialization (before `self.net` assigned)
2. Loading with mismatched Kraken versions

**Workaround**: `ketos convert` best checkpoint to `.safetensors`. If best is
corrupted, use `checkpoint_abort.ckpt` (auto-saved on crash).

**msocr already applies an idempotent in-process patch on the RunPod pod after
pip install** — see `msocr/training/orchestrator.py`.

### 10.3 Other gotchas

- **Distribute weights, not checkpoints**: `.ckpt` files can execute arbitrary
  code. Ship `.safetensors`/`.mlmodel`.
- **`--resize new` required** for alphabet mismatch — without it, fine-tune
  fails if base alphabet ≠ training alphabet.
- **Binary dataset format** is 10–15× faster than XML. Pre-compile with
  `ketos compile`.
- **`--preload`**: datasets <2500 lines preloaded into RAM by default. Disable
  with `--no-preload` for large sets.
- **Validation split**: default 10% held-out. Override with `--partition 0.9`
  or explicit `-e` evaluation files.
- **Baseline-only XML now included**: 7.0 no longer filters lines lacking
  boundary polygons but having baselines. Generally good (more data) but
  msocr's polygon enrichment via `calculate_polygonal_environment` before
  upload remains correct — explicit polygons give better training signal.

---

## 11. Config Review — `msocr/configs/sogdian_config.yaml`

Cross-checking the current config against the research. Four flags fight the
evidence; two are worth tuning.

### 11.1 Mismatches (recommend change)

| Setting | Current | Research says | Why |
|---------|---------|---------------|-----|
| `model.normalization` | `NFC` | **`NFD`** | Syriac/Sogdian has vocalization + diacritics (seyame, vowels, shewa). NFD makes each a separate CTC label → better diacritic accuracy. NFC collapses them. |
| `training.weight_decay` | `0.0001` | **`0`** | CTC-trained RNNs don't benefit from weight decay; Kraken rec default is 0. 1e-5 is for seg only. |
| `training.schedule` | `1cycle` | **`cosine`** | 1cycle is for from-scratch. Christian Sogdian is fine-tuning from `sophro_mhiro_syriac` base → cosine + warmup 200 stabilizes fine-tune. |
| `preprocessing.binarize` | `true` (otsu) | **`false`** | Kraken reads grayscale; binarization destroys faint ink. Damaged manuscripts suffer most. |

### 11.2 Worth tuning

| Setting | Current | Research says | Why |
|---------|---------|---------------|-----|
| `training.lrate` | `0.001` (key was `learning_rate` — never read by `ketos_trainer.py`, so ketos used its default 0.001) | **`0.0001`** (1e-4) for fine-tune; key must be `lrate` to match `_train_subcmd_flags` | 1e-3 is the from-scratch default; fine-tuning needs lower LR to avoid catastrophic forgetting. Renaming the key is also a bugfix — old key was silently ignored. |
| `preprocessing.enhance_contrast` (CLAHE) | `true` | **`false`** | TrOCR ablation 2026: CLAHE not statistically beneficial. Kraken's `--augment` already covers lighting variation via OpticalDistortion + brightness. |

### 11.3 Already correct

- `model.spec`: matches Kraken 7.0 default — correct for Sogdian diacritics
- `model.base_dir: "R"`: correct for RTL Syriac/Sogdian
- `training.optimizer: "Adam"`: correct
- `training.epochs: 100, min_epochs: 20, lag: 10`: reasonable with `--quit early`
- `training.batch_size: 16`: correct for 2000–20k line range
- `training.augment: true`: correct
- `training.precision: "bf16-mixed"`: correct for RTX 30xx+
- `preprocessing.target_height: 120`: matches VGSL spec
- `output.save_best_only: true`: correct

### 11.4 Proposed patch

```yaml
model:
  normalization: "NFD"        # was NFC — diacritics as separate CTC labels

training:
  lrate: 0.0001             # was learning_rate: 0.001 — key renamed (was never read) + value lowered for fine-tune
  weight_decay: 0             # was 0.0001 — CTC rec doesn't use WD
  schedule: "cosine"          # was 1cycle — cosine for fine-tuning
  warmup: 200                 # new — required with cosine fine-tune

preprocessing:
  binarize: false             # was true/otsu — destroys faint ink
  enhance_contrast: false     # was true — CLAHE not beneficial per TrOCR ablation
```

---

## 12. Recommended Training Commands

### 12.1 Christian Sogdian fine-tune (primary path)

```bash
ketos --workers 4 -d cuda:0 --precision bf16-mixed train \
  -i models/kraken/sophro_mhiro_syriac.mlmodel --resize new \
  -f binary -B 16 \
  --base-dir R --normalization NFD \
  --min-epochs 50 --lag 15 --quit early \
  -r 1e-4 --schedule cosine --warmup 200 \
  --augment \
  -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]' \
  christian_sogdian_c2av.arrow
```

### 12.2 Sogdian national from scratch (if Avestan base unavailable)

```bash
ketos --workers 4 -d cuda:0 --precision bf16-mixed train \
  -f binary -B 16 \
  --base-dir R --normalization NFD \
  --min-epochs 20 --lag 10 --quit early \
  -r 1e-3 --schedule 1cycle \
  --augment \
  -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]' \
  sogdian_national.arrow
```

### 12.3 Segmentation fine-tune for fragmentary pages

```bash
ketos segtrain -d cuda:0 \
  -i blla.mlmodel --resize new \
  -f page -t train.txt -e val.txt \
  --suppress-regions \
  --quit fixed --epochs 50 \
  -r 2e-4 --schedule cosine --warmup 200 \
  --augment \
  -o sogdian_fragmentary_seg
```

### 12.4 Evaluate

```bash
ketos test -m model_best.safetensors -e test_manifest.txt -f binary \
  --base-dir R --normalization NFD --fixed-splits
```

---

## 13. Open Problems

1. **No lacuna token for CTC** — fundamental limitation. TrOCR has one; Kraken
   doesn't and can't without architectural change. Split-baseline annotation is
   the workaround.
2. **No published end-to-end CER for fragmentary HTR** — DSS, papyri, palimpsest
   work stops at segmentation or image reconstruction. The biggest gap.
3. **CATMuS is Latin-only** — no equivalent annotation standard for Semitic
   scripts. Diplomatic vs. normalized debate unresolved for non-Latin.
4. **RTL × fragmentary interaction unstudied** — broken baselines on RTL lines
   may confuse reading-order heuristics. Orli's learned ordering may help.
5. **Orli unvalidated on fragments** — Bézier regression *should* handle gaps
   better than heatmaps, but no published numbers yet.
6. **Seg fine-tuning numbers are anecdotal** — "5–20 pages, 40–50 epochs" is
   community consensus, not benchmarked on fragmentary corpora.

---

## References

### Kraken primary
1. Kraken VGSL docs — https://kraken.re/5.1/vgsl.html
2. Kraken training docs — https://kraken.re/6.0.0/training/rectrain.html
3. `default_specs.py` — https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/default_specs.py
4. Augmentation source — https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/recognition.py
5. Kraken 7.0 release — https://github.com/mittagessen/kraken/releases/tag/7.0
6. Kiessling, "Kraken 4" (pretraining) — https://hal.science/hal-05438436v1/file/kraken_ben.pdf
7. Kiessling, "Kraken v5" — https://inria.hal.science/hal-05144723

### Kraken GitHub issues
8. #711 (dataset size) — https://github.com/mittagessen/kraken/issues/711
9. #673 (augmentation fixes) — https://github.com/mittagessen/kraken/pull/673
10. #773 (polygon offset regression) — https://github.com/mittagessen/kraken/issues/773
11. #768 (segtrain crash) — https://github.com/mittagessen/kraken/issues/768
12. #745 (BLLA excessive splitting) — https://github.com/mittagessen/kraken/issues/745
13. #677 (broken lines, UBMA fix) — https://github.com/mittagessen/kraken/issues/677
14. #544 (bottom polygon truncation) — https://github.com/mittagessen/kraken/issues/544

### Fragmented manuscript HTR
15. Borkar & Smith, "Mind the Gap" (TrOCR lacuna) — https://arxiv.org/abs/2407.00250
16. Vogler et al., "Lacuna Reconstruction" (pretrain) — https://arxiv.org/abs/2112.08692
17. Brown-deVost et al., DSS fragment seg — https://arxiv.org/abs/2406.15692
18. Kurar-Barakat & Dershowitz, DSS MTEM — https://arxiv.org/abs/2411.10668
19. TrOCR augmentation ablation — https://arxiv.org/html/2606.24302
20. TrOCR augmentation ensemble — https://arxiv.org/abs/2508.11499
21. HATFormer (Arabic HTR) — https://arxiv.org/abs/2410.02179
22. Palimpsest synthetic data + GAN — MDPI Mathematics 13(14):2304, 2025
23. MsBERT (Hebrew lacuna LM) — ACL 2024 ML4AL Workshop
24. Orli (Bézier line predictor) — https://arxiv.org/abs/2606.04166
25. Orli repo — https://github.com/mittagessen/orli

### Annotation standards
26. CATMuS guidelines — https://catmus-guidelines.github.io/
27. TEI `<gap>` — https://tei-c.org/release/doc/tei-p5-doc/en/html/ref-gap.html
28. EpiDoc lacunae — https://epidoc.stoa.org/gl/latest/trans-lostcharknown.html
29. Clérice & Pinche, "Pre-Editorial Normalization" — arXiv:2602.13905, 2026

### Community guides
30. Digital Orientalist training — https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/
31. eScriptorium training — https://ub-mannheim.github.io/eScriptorium_Dokumentation/Training-with-eScriptorium-EN.html
32. OpenITI ACDC pipeline — https://github.com/OpenITI/acdc_train

### Base models
33. CATMuS Medieval 1.6.0 — https://zenodo.org/records/15030337
34. openiti-arabic-base — https://zenodo.org/records/7050270
35. Avestan (Nikyek) — https://huggingface.co/Nikyek/avestan-ocr-kraken-v1