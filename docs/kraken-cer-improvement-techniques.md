# Kraken HTR CER Improvement Techniques for Low-Data Fine-Tuning

**Context**: 148 lines of Christian Sogdian C2AV plates, fine-tuning from Sophro Mhiro Syriac base (36-char Syriac codec) to 24-char Latin transliteration target. Single manuscript. Current CER ~82% holdout (frozen backbone) / ~91% holdout (unfrozen, with 95% train / 87% val indicating catastrophic forgetting + overfitting).

**Scope**: Every Kraken-side technique (v7.0+) that could reduce CER without new annotations. Techniques requiring new data are noted but deprioritized.

---

## 1. Learning Rate Schedules

### What Kraken supports

Kraken v7.0+ supports these schedules via `--schedule`:

| Schedule | CLI flag | Key params | Best for |
|----------|----------|------------|----------|
| `constant` | `--schedule constant` | `-r` (default 1e-3) | Baseline, stable datasets |
| `cosine` | `--schedule cosine` | `--cos-t-max`, `--cos-min-lr` | Fine-tuning, avoids overshoot |
| `1cycle` | `--schedule 1cycle` | Cycle length = `--epochs` | Small datasets, fast convergence |
| `exponential` | `--schedule exponential` | `-g` (gamma), `--step-size` | Gradual decay |
| `step` | `--schedule step` | `--step-size`, `-g` | Staged drops |
| `reduceonplateau` | `--schedule reduceonplateau` | `--sched-patience` | Adaptive, but slow to react |
| `cosine_warm_restarts` | `--schedule cosine_warm_restarts` | `--cos-t-max` | Cyclic, good for escaping local minima |

All schedules support `--warmup N` (linear ramp-up over N steps).

**Source**: [Kraken Training Recognition docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst)

### What matters for our case

- **Default `constant` at 1e-3 is too aggressive** for 148 lines. The unfrozen run (95% train / 87% val) shows the model is memorizing training data.
- **Lower LR is critical**: The Digital Orientalist tutorial reports better results with `-r 1e-4` for connected scripts like Nastaliq. The Kraken docs recommend scaling LR by `sqrt(batch_size)` when changing batch size.
- **Cosine with warmup** is the community-preferred schedule for fine-tuning. The Kraken segmentation training example uses `schedule: cosine` with `cos_t_max: 50`, `cos_min_lr: 2e-5`, `warmup: 200`.
- **1cycle** is theoretically ideal for small datasets (one pass up, one pass down) but the Kraken docs note it's "slower to train and often requires a couple of epochs to output any sensible text."

**Source**: [Digital Orientalist Kraken tutorial](https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/), [Kraken rectrain docs](https://kraken.re/6.0.0/training/rectrain.html)

### Recommendation for our case

```
--schedule cosine --cos-t-max 50 --cos-min-lr 1e-5 -r 5e-5 --warmup 500
```

**Rationale**: Very low peak LR (5e-5) to prevent catastrophic forgetting. Cosine decay to 1e-5. Long warmup (500 steps) to let the new output head stabilize before the backbone starts moving. This is the single highest-impact hyperparameter change.

**Expected CER impact**: **HIGH**. The 95% train / 87% val gap is a classic LR-too-high signature.

---

## 2. Backbone Freeze Strategies

### What `--freeze-backbone` does

The `--freeze-backbone N` flag freezes all layers except the output (classification) head for the first **N training samples** (not iterations/epochs). After N samples, the backbone unfreezes and trains normally.

**Source**: [Kraken fine-tuning docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst)

### Our results analyzed

| Strategy | Train CER | Val CER | Holdout CER | Diagnosis |
|----------|-----------|---------|-------------|-----------|
| `--freeze-backbone 5000` | ~82% | ~82% | ~82% | Head never learned mapping. 5000 samples ≈ 35 epochs frozen — too long for 148 lines. |
| No freeze (unfrozen) | 95% | 87% | 91% | Catastrophic forgetting + overfitting. Backbone overwrote Syriac features. |

### What we should try

**Intermediate freeze values**: The Kraken docs show `--freeze-backbone 1000` for pretrained model fine-tuning. For 148 lines with batch size 32, that's ~31 batches frozen. Try:

- `--freeze-backbone 200` (~6 batches, ~1.3 epochs): Just enough to let the new output head initialize
- `--freeze-backbone 500` (~16 batches, ~3.4 epochs): Let head stabilize before backbone moves
- `--freeze-backbone 1000` (~31 batches, ~6.8 epochs): More conservative

**Progressive unfreezing**: Kraken does not natively support progressive unfreezing (unfreeze layer by layer). However, you can simulate it:
1. Train with `--freeze-backbone 1000` → save checkpoint
2. Resume with `--resume checkpoint.ckpt` and no freeze → backbone trains from there

**Partial layer freezing**: Kraken does not support freezing specific layers via CLI. The `--append` slicing mechanism (see §8) can achieve a similar effect by discarding upper layers and retraining them.

**Source**: [Kraken Issue #711](https://github.com/mittagessen/kraken/issues/711) (maintainer used `--freeze-backbone` with fine-tuning), [TrOCR ablation study](https://arxiv.org/html/2606.24302v1) (encoder freezing is fragile, decoder freezing is safer — analogous to Kraken's CNN backbone vs. RNN head)

### Recommendation for our case

```
--freeze-backbone 500
```

**Rationale**: 500 samples (~3.4 epochs at batch 32) lets the new 24-char output head learn the Latin transliteration mapping while the Syriac visual features stay intact. Then unfreeze gently with low LR.

**Expected CER impact**: **HIGH**. The frozen 5000 run proved the head needs training; the unfrozen run proved the backbone needs protection. 500 is the Goldilocks zone.

---

## 3. Augmentation (`--augment`)

### What Kraken applies

Kraken's `DefaultAugmenter` (in `kraken/lib/dataset/recognition.py`) applies these transforms via `albumentations`, with 50% probability per image:

| Transform | Probability | Purpose |
|-----------|------------|---------|
| `PixelDropout` | 20% | Random pixel zeroing (simulates ink degradation) |
| `MotionBlur` | 20% | Directional blur (camera movement) |
| `MedianBlur` (kernel ≤3) | 10% | Median filtering (ink bleed) |
| `Blur` (kernel ≤3) | 10% | Gaussian blur (out-of-focus) |
| `OpticalDistortion` | 30% | Lens/curvature distortion |
| `ElasticTransform` (α=7, σ=25) | 10% | Parchment warping |
| `SafeRotate` (±3°) | 20% | Slight rotation |

**Source**: [Kraken DefaultAugmenter source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/recognition.py)

### When augmentation helps vs. hurts

- **Helps**: Small datasets, single-manuscript training (our case). Prevents the model from memorizing exact pixel patterns.
- **Hurts**: When the base model already generalizes well and augmentation introduces unrealistic distortions. The Kraken maintainer used `--augment` in both the 700-line and 1000-line fine-tuning examples (Issue #711).
- **TrOCR study finding**: Augmentation is dataset-dependent — beneficial on READ-16, neutral on Cortonese. Must validate per dataset.

**Source**: [TrOCR ablation study](https://arxiv.org/html/2606.24302v1), [Kraken Issue #711](https://github.com/mittagessen/kraken/issues/711)

### Tunability

Kraken's built-in augmentation is **not tunable** via CLI. The probabilities and parameters are hardcoded in `DefaultAugmenter`. To customize, you'd need to subclass the dataset in Python.

### Recommendation for our case

```
--augment
```

**Always on for 148 lines.** The single-manuscript overfitting risk is high. Augmentation is the cheapest regularizer available.

**Expected CER impact**: **MEDIUM**. Helps with overfitting but won't fix the fundamental codec-mapping problem.

---

## 4. Data Augmentation Beyond `--augment`

### Kraken-native options

**Pretraining on unlabeled lines** (`ketos pretrain`): Kraken supports wav2vec2-style contrastive pretraining on unlabeled line images. This learns visual representations without transcriptions. Then fine-tune on the 148 labeled lines.

```bash
ketos pretrain -t unlabeled_lines.lst -o pretrain_checkpoints
ketos train -i pretrain_best.safetensors --warmup 5000 --freeze-backbone 1000 -f binary labelled.arrow
```

**Key**: This requires additional unlabeled line images from the same manuscript (or similar material). If you have more C2AV plate images without transcriptions, this is high-leverage.

**Source**: [Kraken pretraining docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst), [HAL paper on Kraken pretraining](https://hal.science/hal-05438436v1/file/kraken_ben.pdf) (claims "30 lines for supervised fine-tuning" after pretraining)

### Community tools

- **Synthetic data generation**: No Kraken-native tool. External options: TextRender, TRDG, or custom Pillow-based rendering of Sogdian text in Syriac-style fonts (if any exist).
- **Albumentations customization**: Install `albumentations` and modify the `DefaultAugmenter` class to add stronger transforms (e.g., `GridDistortion`, `CoarseDropout`, `CLAHE`).

### Recommendation for our case

**If you have unlabeled C2AV plate images**: Run `ketos pretrain` on them, then fine-tune. This is the highest-leverage technique that doesn't require new transcriptions.

**If not**: Skip. Synthetic data generation for Sogdian in Syriac script is a research project in itself.

**Expected CER impact**: **HIGH** (if unlabeled data available), **N/A** (otherwise).

---

## 5. Preprocessing / Normalization

### What Kraken applies internally

Kraken applies these transforms during training data loading:

- **Unicode normalization**: `--normalization NFD` (or NFC, NFKD, NFKC). Default is `NFD`. This decomposes combined characters.
- **Whitespace normalization**: `--normalize-whitespace` (default: true). Collapses multiple spaces.
- **Binarization**: NOT applied by default during training. Kraken trains on grayscale (or color if specified in VGSL spec). The `--force-binarization` flag exists but is for legacy models.
- **Image transforms**: Rescaling to match the network's expected input dimensions (from VGSL spec).

**Source**: [Kraken training recognition docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst), [Kraken binarization API](https://github.com/mittagessen/kraken/blob/main/kraken/kraken/binarization.py)

### What we can control

- **`--normalization NFC`**: For Latin transliteration output, NFC (composed) may be more appropriate than NFD (decomposed). Latin characters with diacritics (š, ž, ʾ, γ, θ) may behave differently under NFD vs NFC.
- **Pre-binarization**: Apply `kraken binarize` to plate images before training. The `nlbin` algorithm has tunable parameters (`--threshold`, `--zoom`, `--perc`, `--range`, `--low`, `--high`). Cleaner input → better training signal.
- **Deskew/denoise**: Not built into Kraken training pipeline. Must be done externally (e.g., ImageMagick, OpenCV) before creating XMLs.

### Recommendation for our case

```bash
# Pre-binarize plates before training
kraken -i plate.png plate_bin.png binarize --threshold 0.5
```

Then regenerate XMLs from binarized images. Also try `--normalization NFC` since our output is Latin.

**Expected CER impact**: **LOW-MEDIUM**. Binarization helps if plates have significant background noise/variance. For clean manuscript photos, impact is marginal.

---

## 6. Line Segmentation Quality

### How it affects training

Kraken extracts line images from PAGE XML using either:
- **Baseline + polygon**: The polygon defines the exact pixel region of the line. Kraken v7 uses a new polygon extractor (not legacy).
- **Baseline only**: Kraken auto-generates polygons around the baseline.

**Critical**: The polygon determines which pixels the recognition model sees during training. If polygons are too tight (cutting off ascenders/descenders) or too loose (including neighboring lines), the model learns from noisy input.

### Our situation

Our XMLs are baseline-only. We auto-enrich with polygons via `kraken.lib.segmentation`. The quality of auto-generated polygons depends on:
- Baseline accuracy (are baselines correctly placed through the text body?)
- Line spacing (tightly spaced lines → polygons may overlap)
- The polygon extraction method (legacy vs. new in v7)

### What we can do

1. **Inspect auto-generated polygons**: Extract line images and visually check if they correctly capture the full text line.
2. **Manual polygon correction**: In eScriptorium, manually adjust polygon boundaries for a subset of lines.
3. **Use `--repolygonize`**: Forces Kraken to regenerate polygons during training data loading.
4. **Check legacy vs. new extractor**: Kraken v7 logs whether it's using legacy or new polygon extraction. The new extractor is generally better.

**Source**: [Kraken segmentation training docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_segmentation.rst)

### Recommendation for our case

Visually inspect 10-20 extracted line images. If polygons are cutting off text or including neighbors, fix the worst offenders manually. For 148 lines, even 10 bad polygons is ~7% noise in the training signal.

**Expected CER impact**: **LOW-MEDIUM**. Only matters if polygons are systematically bad. Quick visual check first.

---

## 7. Recognition Model Architecture (`--spec`)

### Default architecture

Kraken's default recognition spec:
```
[1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]
```

This is: input [1,48,0,1] (batch, height=48, variable width, 1 channel grayscale), then 2 conv blocks with dropout and maxpool, a reshape layer, and an LSTM + linear output.

**Source**: [Kraken VGSL docs](https://kraken.re/5.2/vgsl.html), [Kraken recognition.py source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/ketos/recognition.py)

### Architecture options for low-data

| Strategy | Spec change | Rationale |
|----------|-------------|-----------|
| **Smaller model** | Reduce filters: `Cr3,3,16` instead of `Cr3,3,32` | Fewer params → less overfitting |
| **More dropout** | `Do0.3,2` instead of `Do0.1,2` | Stronger regularization |
| **Shallower** | Remove one conv block | Fewer layers → less capacity to memorize |
| **Smaller LSTM** | `Lbx50` instead of `Lbx100` | Smaller recurrent layer |
| **Color input** | `[1,48,0,3]` (3 channels) | If plates are color, preserves more information |

### The `--append` slicing mechanism

When the codec mismatch is extreme (36 Syriac chars → 24 Latin chars), Kraken docs recommend **slicing**: discard the top layers and retrain them from scratch while keeping the CNN backbone:

```bash
# Slice off everything after layer 7 (the conv stack) and add new RNN layers
ketos train -i sophro_mhiro_syriac.safetensors --append 7 \
  -s '[Lbx128 Do0.1,2 Lbx128 Do0.1,2]' --resize new training_data/*.xml
```

This preserves the Syriac visual feature extractor (which knows what Syriac-script letterforms look like) but retrains the sequence modeling layers from scratch for the new codec.

**Source**: [Kraken slicing docs](https://kraken.re/5.3.0/ketos.html#slicing), [Kraken ketos.rst](https://github.com/mittagessen/kraken/blob/9a218ce8/docs/ketos.rst)

### Recommendation for our case

**Try slicing before changing architecture.** The `--append 7` approach is specifically designed for our situation (highly different alphabets). It's the Kraken-recommended solution for "the modification of the final linear layer to add/remove character will destroy the inference capabilities of the network."

```bash
ketos train -i sophro_mhiro_syriac.safetensors --append 7 \
  -s '[Lbx128 Do0.1,2 Lbx128 Do0.1,2]' --resize new \
  --freeze-backbone 0 --augment \
  --schedule cosine -r 1e-4 --warmup 200 \
  training_data/*.xml
```

Note: `--freeze-backbone 0` because `--append 7` already discards the old RNN layers. The CNN backbone trains from step 0 but with fresh RNN layers on top.

**Expected CER impact**: **HIGH**. This is the most architecturally appropriate approach for our codec-mismatch situation.

---

## 8. Training Data Ordering / Sampling

### What Kraken does

- **Shuffling**: Training lines are shuffled each epoch (standard PyTorch DataLoader behavior).
- **No weighting/oversampling**: Kraken does not support per-line weights or oversampling of rare characters.
- **Batch composition**: Random sampling. No class-balancing or hard-example mining built in.
- **Preloading**: Datasets with <2500 lines are preloaded into memory by default (`--preload`). Can be disabled with `--no-preload`.

**Source**: [Kraken training docs](https://kraken.re/6.0.0/tutorials/training.html)

### What we can do

- **Manual oversampling**: Duplicate lines with rare characters in the training manifest. If certain Latin characters (ž, ʾ, γ, θ) appear infrequently, duplicate those lines 2-3x.
- **Fixed splits**: Use explicit manifest files (`train.lst`, `val.lst`) instead of random splits. This ensures reproducibility and lets you control which plates go where.
- **Plate-level stratification**: Ensure each split has lines from different plates (not random line-level splitting which can put lines from the same plate in both train and val).

### Recommendation for our case

Create explicit manifests with plate-level splitting. Count character frequencies and duplicate lines containing rare characters (appearing <5 times).

**Expected CER impact**: **LOW**. Helps at the margins but won't fix the fundamental data scarcity.

---

## 9. Validation Strategy

### The problem

Our 15-line val plate is too small — the val CER is noisy and early stopping triggers too aggressively (or not at all). The 87% val / 91% holdout gap in the unfrozen run suggests the val set is not representative.

### What Kraken supports

| Parameter | Flag | Effect |
|-----------|------|--------|
| Validation frequency | `-F 0.5` | Validate every half-epoch (more frequent feedback) |
| Early stopping lag | `--lag 10` | Wait 10 evaluations before stopping (default) |
| Minimum delta | `--min-delta 0.001` | Minimum improvement to reset early stopping |
| Minimum epochs | `--min-epochs 30` | Don't stop before epoch 30 |
| Fixed epochs | `--quit fixed --epochs 50` | Disable early stopping entirely |
| Explicit val set | `-e val.lst` | Use fixed validation manifest |

**Source**: [Kraken training recognition docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst)

### Strategies for small val sets

1. **Larger val set**: Combine val + holdout into a single ~30-line val set. Accept that you lose the independent holdout.
2. **Leave-one-plate-out**: Train on 11 plates, validate on 1, rotate. Average results. Not natively supported by ketos but can be scripted.
3. **Disable early stopping**: Use `--quit fixed --epochs 50` and manually inspect the checkpoint CERs afterward. Pick the best.
4. **More frequent validation**: `-F 0.25` (validate 4x per epoch) gives more data points for early stopping, but each validation pass is on the same 15 lines.

### Recommendation for our case

```
--quit fixed --epochs 50 -F 0.5
```

Disable early stopping. Train for 50 epochs. Save all checkpoints. After training, run `ketos test` on the holdout set against each checkpoint and pick the best. This avoids the noisy-val early-stopping problem entirely.

**Expected CER impact**: **MEDIUM**. Prevents premature stopping, which is likely happening with 15 val lines.

---

## 10. Loss Functions / Objectives

### What Kraken uses

Kraken recognition models use **CTC (Connectionist Temporal Classification)** loss exclusively. There is no alternative loss function (no cross-entropy, no sequence-to-sequence with attention).

The output layer is automatically appended as `O1c{N}` where N = number of codec classes + 1 (blank for CTC).

**Source**: [Kraken VGSL docs](https://kraken.re/5.2/vgsl.html) ("When training sequence classification networks with the provided tools the appropriate output definition is automatically appended")

### What we can control

- **Gradient clipping**: `--gradient-clip-val 1.0` (prevents exploding gradients, useful for unstable fine-tuning)
- **Gradient accumulation**: `--accumulate-grad-batches 2` (effective batch size = batch_size × accumulation steps, useful when GPU memory limits batch size)
- **Weight decay**: `-w 1e-5` (L2 regularization, helps prevent overfitting)

### Recommendation for our case

```
--gradient-clip-val 1.0 -w 1e-4
```

Weight decay is underused in Kraken fine-tuning. At 148 lines, L2 regularization is cheap insurance against overfitting.

**Expected CER impact**: **LOW**. CTC is the only option. Weight decay and gradient clipping are stability improvements, not CER game-changers.

---

## 11. Pretrained Model Selection

### Our base: Sophro Mhiro Syriac

- **Training data**: 3,466 lines, 90,991 chars, 38 manuscripts, 100 images
- **Codec**: 36 Syriac characters (consonants + limited punctuation/diacritics)
- **Accuracy**: 97.4% on test set
- **Zenodo**: [10.5281/zenodo.17406773](https://doi.org/10.5281/zenodo.17406773)

### Alternative bases

| Model | Script | Codec size | Lines | Relevance |
|-------|--------|------------|-------|-----------|
| **Sophro Mhiro Syriac** | Syriac (Serto/Estrangela/Eastern) | 36 | 3,466 | **Current. Best visual match for Sogdian in Syriac script.** |
| CATMuS Medieval 1.6 | Latin manuscripts | Large | 100K+ | Wrong script family. Visual features don't transfer. |
| MiDRASH Geniza 01 | Hebrew/Aramaic/Judeo-Arabic | Large | Unknown | Hebrew square script — different from Syriac cursive. |
| Kraken default Arabic | Arabic | ~40 | Unknown | Arabic script is cursive like Syriac but letterforms differ. |
| dstoekl/kraken_models Syriac | Syriac print + manuscript | Varies | Unknown | Alternative Syriac models, may have different training data. |

**Source**: [dstoekl/kraken_models](https://github.com/dstoekl/kraken_models), [MiDRASH Geniza](https://zenodo.org/records/17931017), [CATMuS](https://zenodo.org/records/15030337)

### Analysis

The Sophro Mhiro base is **the best available option** for Sogdian in Syriac script. Christian Sogdian uses the Syriac alphabet with a few additional letters. The visual backbone has seen Syriac letterforms across 38 manuscripts — this is exactly what we need.

The problem is not the base model choice; it's the **codec mapping** (Syriac Unicode → Latin transliteration). No public model maps Syriac-script images to Latin output. We must rebuild the output head regardless of which base we choose.

### Recommendation for our case

**Stick with Sophro Mhiro Syriac.** No better base exists for Syriac-script manuscript recognition. The codec mismatch is unavoidable — focus on the slicing approach (§7) to handle it properly.

**Expected CER impact**: **N/A** (no better alternative exists).

---

## 12. Ensembling / Model Averaging

### What Kraken supports

- **No native ensembling**: `ketos` does not support combining multiple models at inference time.
- **No SWA (Stochastic Weight Averaging)**: Not implemented in Kraken's training loop.
- **Checkpoint averaging**: You can manually average the weights of multiple checkpoints using Python/PyTorch, but this is not a built-in feature.

### What we can do

- **Manual checkpoint selection**: Train multiple runs with different seeds/hyperparams. Run `ketos test` on holdout for each. Pick the best single model.
- **Post-hoc averaging**: Load the state dicts of the top-3 checkpoints, average them, save as new model. Requires Python scripting.

### Recommendation for our case

**Skip for now.** At 148 lines, the variance between runs is dominated by data scarcity, not model initialization. Ensembling won't help until the single-model CER is much lower.

**Expected CER impact**: **LOW**. Premature optimization at 82% CER.

---

## 13. Hyperparameter Search

### Community workflows

No formal hyperparameter search tooling exists for Kraken. The community pattern is manual grid search:

1. Fix a validation strategy (e.g., leave-one-plate-out)
2. Sweep over: learning rate, freeze-backbone, warmup, batch size
3. Compare holdout CER

### Recommended search ranges for 148-line fine-tune

| Parameter | Range | Step |
|-----------|-------|------|
| `-r` (learning rate) | 1e-5 to 1e-3 | ×3 (1e-5, 3e-5, 1e-4, 3e-4, 1e-3) |
| `--freeze-backbone` | 0, 200, 500, 1000, 2000 | Discrete |
| `--warmup` | 100, 200, 500, 1000 | Discrete |
| `-B` (batch size) | 8, 16, 32 | Discrete |
| `--schedule` | constant, cosine, 1cycle | Discrete |

**Source**: [Kraken Issue #711](https://github.com/mittagessen/kraken/issues/711) (maintainer's hyperparameter choices), [Digital Orientalist](https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/) (community LR recommendations)

### Recommendation for our case

Run a focused grid over the three highest-impact parameters:

```bash
for lr in 1e-5 5e-5 1e-4; do
  for freeze in 200 500 1000; do
    ketos train --load sophro.safetensors --resize new \
      -r $lr --freeze-backbone $freeze --warmup 500 \
      --schedule cosine --augment --quit fixed --epochs 50 \
      -t train.lst -e val.lst
    ketos test -m *_best.safetensors -e holdout.lst
  done
done
```

**Expected CER impact**: **MEDIUM**. Won't create miracles but will find the best combination of the techniques above.

---

## 14. Curriculum Learning

### Kraken support

**None.** Kraken does not support easy-to-hard ordering of training samples. Lines are shuffled randomly each epoch.

### What we could do

- **Manual curriculum**: Sort training lines by length (shortest first). Train for N epochs on short lines, then add longer lines. Requires multiple training runs with different manifests.
- **Length-based batching**: Not supported by Kraken. Would require modifying the DataLoader.

### Recommendation for our case

**Skip.** The implementation effort is high and the benefit for 148 lines is speculative. Curriculum learning helps most when there's a wide difficulty range — our 12 plates from one manuscript likely have uniform difficulty.

**Expected CER impact**: **LOW**. Not worth the complexity at this scale.

---

## 15. Active Learning / Hard-Example Mining

### Kraken-native tools

- **`ketos test`**: Produces per-character confusion matrices and per-script accuracy. Does NOT produce per-line CER.
- **No built-in active learning loop**: Kraken does not identify which lines to re-annotate.

### Community tools

- **FoNDUE-HTR/Hands_clustering**: An external tool that clusters pages by handwriting style, computes per-page CER via `ketos test`, and ranks untranscribed pages by estimated difficulty. Designed for prioritizing annotation effort, not for improving an existing model.

**Source**: [Hands_clustering GitHub](https://github.com/FoNDUE-HTR/Hands_clustering)

### What we can do

- **Manual error analysis**: Run `ketos test -m model.safetensors -e holdout.lst`. Examine the confusion matrix. Identify which characters are systematically confused (e.g., š vs. s, ž vs. z). Check if those characters are underrepresented in training.
- **Re-annotate high-error lines**: If specific lines have outlier CER, re-check their transcriptions for errors.
- **Character frequency analysis**: Count occurrences of each of the 24 Latin characters in the 148 training lines. Characters with <5 examples will never be learned.

### Recommendation for our case

Run `ketos test` on the holdout set. If certain characters have 0% accuracy, those characters need more training examples. This is likely the case for rare characters like ʾ, γ, θ, ž.

**Expected CER impact**: **MEDIUM** (if transcription errors found), **LOW** (if data is clean but scarce).

---

## 16. CER-Specific Inference Tricks

### What Kraken supports at inference time

| Option | Flag | Effect |
|--------|------|--------|
| Temperature | `--temperature 0.5` | Sharpens probability distribution (lower = more confident, less diverse output) |
| Bidi reordering | `-d rtl` | Forces right-to-left text direction |
| No segmentation | `--no-segmentation` | Use pre-cropped line images |
| Unicode normalization | `-u NFD` | Normalize output text |
| Model format | `.safetensors` vs `.mlmodel` | safetensors is v7 native, mlmodel is legacy |

**Source**: [Kraken inference docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/inference.rst)

### What Kraken does NOT support

- **No language model / decoder**: Kraken uses greedy CTC decoding. No beam search, no lexicon, no language model rescoring.
- **No post-correction**: No built-in spell-checking or ByT5-style post-processing.
- **No confidence thresholding**: Cannot filter low-confidence characters (though the Python API exposes per-character confidence).

### External post-correction

The [Modular Pipeline paper](https://doi.org/10.3390/electronics14153083) demonstrates fine-tuning a ByT5 model on Kraken output to correct OCR errors. They achieved CER reduction from Kraken's raw output using <500 training pairs. This is a separate step after Kraken inference.

### Recommendation for our case

- **`--temperature 0.3`**: At inference, lower temperature may help by forcing the model to commit to its best guess rather than hedging. Test on holdout.
- **Post-correction with ByT5**: Only worth exploring if Kraken CER drops below ~30%. At 82%, the output is too noisy for a correction model to learn meaningful patterns.

**Expected CER impact**: **LOW** (temperature tuning), **MEDIUM** (post-correction, but only after CER improves).

---

## Ranked Recommendations for Our Case

| # | Technique | CER Impact | Effort | CLI flags / action | Why it matters |
|---|-----------|------------|--------|-------------------|----------------|
| 1 | **Network slicing (`--append`)** | **HIGH** | MEDIUM | `--append 7 -s '[Lbx128 Do0.1,2 Lbx128 Do0.1,2]'` | Codec mismatch (36→24) destroys output head. Slicing is the Kraken-recommended fix. |
| 2 | **Low LR + cosine schedule** | **HIGH** | LOW | `-r 5e-5 --schedule cosine --cos-min-lr 1e-5 --warmup 500` | 95% train / 87% val gap is classic LR-too-high. |
| 3 | **Intermediate freeze-backbone** | **HIGH** | LOW | `--freeze-backbone 500` | 5000 was too long (head starved), 0 caused forgetting. 500 is the sweet spot. |
| 4 | **Disable early stopping** | **MEDIUM** | LOW | `--quit fixed --epochs 50 -F 0.5` | 15-line val set is too noisy for reliable early stopping. |
| 5 | **Pretrain on unlabeled plates** | **HIGH** | MEDIUM | `ketos pretrain` then fine-tune | If you have unlabeled C2AV images, this is the highest-leverage data-side technique. |
| 6 | **Hyperparameter grid search** | **MEDIUM** | MEDIUM | Scripted sweep over lr, freeze, warmup | Find the best combination of techniques 1-4. |
| 7 | **Augmentation** | **MEDIUM** | LOW | `--augment` | Always on for 148 lines. Prevents pixel-level memorization. |
| 8 | **Weight decay** | **LOW-MED** | LOW | `-w 1e-4` | L2 regularization, cheap insurance against overfitting. |
| 9 | **Character frequency analysis** | **MEDIUM** | LOW | `ketos test` confusion matrix | Identify which of the 24 chars are never learned. Duplicate their training lines. |
| 10 | **Polygon quality check** | **LOW-MED** | LOW | Visual inspection of 20 line images | Bad polygons = noisy training signal. Quick check first. |
| 11 | **Pre-binarization** | **LOW** | LOW | `kraken binarize` | Helps if plates have background noise. |
| 12 | **Gradient clipping** | **LOW** | LOW | `--gradient-clip-val 1.0` | Stability improvement. |
| 13 | **Unicode normalization** | **LOW** | LOW | `--normalization NFC` | Latin diacritics may behave better under NFC. |
| 14 | **Inference temperature** | **LOW** | LOW | `--temperature 0.3` | Test on holdout after training. |
| 15 | **Manual oversampling** | **LOW** | LOW | Duplicate rare-char lines in manifest | Helps if character distribution is very skewed. |
| 16 | **Alternative base model** | **N/A** | N/A | N/A | No better base exists for Syriac-script Sogdian. |
| 17 | **Ensembling** | **LOW** | HIGH | Manual weight averaging | Premature at 82% CER. |
| 18 | **Curriculum learning** | **LOW** | HIGH | Manual manifest ordering | Not worth complexity at 148 lines. |
| 19 | **Post-correction (ByT5)** | **LOW** | HIGH | Separate training pipeline | Only viable if CER drops below ~30%. |

---

## Bottom Line

**The binding constraint is data quantity, not Kraken configuration.** At 148 lines with a complete codec mismatch (Syriac script → Latin output), no hyperparameter tuning will produce a publishable model (CER <10%). The techniques above can plausibly reduce CER from ~82% to the **40-60% range** — usable for bootstrapping, not for production.

**The single highest-ROI action is adding more annotated lines**, especially from different plates or manuscripts. If that's impossible, the slicing + low-LR + intermediate-freeze combination (techniques 1-3) is the best Kraken-side path.

**If you have unlabeled C2AV plate images**, `ketos pretrain` (technique 5) is the closest thing to "more data" without more annotation effort.

---

## Sources

1. [Kraken Training Recognition docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst)
2. [Kraken Recognition Training (v6.0)](https://kraken.re/6.0.0/training/rectrain.html)
3. [Kraken Training Tutorial (v6.0)](https://kraken.re/6.0.0/tutorials/training.html)
4. [Kraken VGSL Specification](https://kraken.re/5.2/vgsl.html)
5. [Kraken ketos.rst (source)](https://github.com/mittagessen/kraken/blob/9a218ce8/docs/ketos.rst)
6. [Kraken DefaultAugmenter source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/recognition.py)
7. [Kraken binarization API](https://github.com/mittagessen/kraken/blob/main/kraken/kraken/binarization.py)
8. [Kraken pretraining docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_recognition.rst)
9. [Kraken inference docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/inference.rst)
10. [Kraken Issue #711 — fine-tuning discussion](https://github.com/mittagessen/kraken/issues/711)
11. [Kraken recognition.py source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/ketos/recognition.py)
12. [Kraken train.py source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/train.py)
13. [Digital Orientalist — Kraken training tutorial](https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/)
14. [Skidzun/kraken_training — community training guide](https://github.com/Skidzun/kraken_training)
15. [Sophro Mhiro Syriac model — Zenodo](https://doi.org/10.5281/zenodo.17406773)
16. [dstoekl/kraken_models — alternative Syriac models](https://github.com/dstoekl/kraken_models)
17. [HAL — New Developments in Kraken (pretraining)](https://hal.science/hal-05438436v1/file/kraken_ben.pdf)
18. [TrOCR Ablation Study (freezing strategies)](https://arxiv.org/html/2606.24302v1)
19. [Pippi et al. — Fine-tuning for HTR](https://doi.org/10.48550/arxiv.2305.02593)
20. [FoNDUE-HTR/Hands_clustering — active learning](https://github.com/FoNDUE-HTR/Hands_clustering)
21. [Modular Pipeline — ByT5 post-correction](https://doi.org/10.3390/electronics14153083)
22. [ETCBC — Syriac HTR assessment](https://etcbc.nl/methodology/htr-for-syriac-manuscripts-an-assessment-as-of-march-2026/)
23. [La Paperista — 71-line fine-tuning case study](https://aishapectyo.com/2026/05/02/first-pass-handwriting-ocr-on-1900s-manuscripts/)
24. [OICEN-HTR — Old Norse fine-tuning results](https://github.com/NKCZ/OICEN-HTR)
25. [Kraken segmentation training docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_segmentation.rst)
26. [Kraken reading order training docs](https://github.com/mittagessen/kraken/blob/main/docs/user_guide/training_reading_order.rst)
27. [Kraken pretrain model source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/pretrain/model.py)