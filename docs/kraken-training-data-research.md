# Kraken HTR from Scratch for Sogdian — Research Report

> Research conducted to answer: **How many line-image/transcription pairs are needed to train a Kraken HTR model from scratch (no base model) for Sogdian manuscripts to reach "generally good" CER?**
>
> Source: deep web research via @librarian, mid-2026. Image attachment `page-19.png` could not be read (text-only model) — research is text-based.

## TL;DR

| Tier | Lines | Expected CER (from scratch) | Verdict |
|---|---|---|---|
| Absolute minimum | ~1,000 | 20–30% | Barely usable, heavy correction |
| **Realistic first target** | **3,000–5,000** | **10–15%** | **Searchable, correctable — genuinely useful** |
| Production quality | 10,000+ | 5–10% | Good for scholarly work |
| Multi-hand general model | 20,000+ | 2–5% | Approaches production |
| Excellent | 50,000+ | <2% | Near-human |

**For Sogdian specifically: aim for 3,000–5,000 lines as the first serious milestone; push to 10,000+ for sub-10% CER.** Before pure from-scratch, finetune from the existing Avestan Kraken model (`avestan_ms0040.mlmodel` in `models/kraken/`) — even an out-of-domain base can cut data requirements by 50%+.

---

## 1. Kraken Official Documentation

The canonical guidance is consistent across Kraken versions 2.0–7.0.

From the [Kraken Training Tutorial](https://kraken.re/6.0.0/tutorials/training.html) (and [training.rst source](https://github.com/mittagessen/kraken/blob/9a218ce8/docs/training.rst)):

> *"A specific recognition model for printed script with a small grapheme inventory such as Arabic or Hebrew requires around **800 lines**, with manuscripts, complex scripts (such as polytonic Greek), and general models for multiple typefaces and hands needing more training data for the same accuracy."*

> *"There is no hard rule for the amount of training data and it may be required to retrain a model after the initial training data proves insufficient. Most western texts contain between 25 and 40 lines per page, therefore **upward of 30 pages** have to be preprocessed and later transcribed."*

From the [Introduction to ATR (v7.0)](https://kraken.re/main/atr.html):

> *"If you only need to recognize a specific manuscript written in a clean, uniform hand with a small alphabetic script, training a model from scratch can be surprisingly fast, often requiring only **a couple of dozen pages** of training data."*

The documentation's example training run shows **788 training lines + 88 validation lines** with 49 unique Unicode code points, and warns:

> *"Characters that occur less than 10 times will most likely not be recognized well by the trained net."*

The `--preload` flag defaults to preloading datasets with **<2,500 lines** into memory — suggesting 2,500 is considered a "small" dataset by Kraken's design.

---

## 2. Published Papers Using Kraken — Dataset Sizes and CER

### A. Medieval Manuscript HTR (Reul, Springmann, Tomasek, Langhanki — 2022)

**Paper**: *"Open Source Handwritten Text Recognition on Medieval Manuscripts using Mixed Models and Document-Specific Finetuning"*
**URL**: [arXiv:2201.07661](https://arxiv.org/abs/2201.07661) / [Springer](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_28)

- Corpus: 35 manuscripts, ~12,500 lines (Gothic and Bastarda cursives, German)
- Mixed model out-of-the-box CER: 6.22%
- After finetuning on 2 pages: 3.27% CER
- After finetuning on 32 pages: 1.65% CER
- **Training from scratch on 32 pages: 5.27% CER (Gothic) / 8.33% CER (Bastarda)**
- Key finding: even an out-of-domain pretrained model as starting point beats training from scratch by a large margin (85% improvement with 2 pages, 39% with 32 pages)

### B. TRIDIS — Medieval Documentary Manuscripts (Torres Aguilar, Jolivet — 2023/2024)

**Paper**: *"Handwritten Text Recognition for Documentary Medieval Manuscripts"*
**URL**: [HAL: hal-03892163](https://hal.science/hal-03892163) / [Zenodo model](https://zenodo.org/records/13862096)

- Training data: 2,950 pages, 245,000 lines, ~2.3M tokens (Latin, Old French, Old Spanish, 11th–16th c.)
- Architecture: Kraken CRNN+CTC
- Validation accuracy: 95.2% (~4.8% CER)
- On 4 unseen external datasets: CER 10–14%
- After finetuning with 10 pages: CER drops to 6–10%
- With GAN-augmented synthetic data (420k lines): CER 5–10%, WER 13–24%

### C. ARletta — Historic Dutch Incident Books (2024)

**Paper**: *"ARletta. Open-Source Handwritten Text Recognition Models for Historic Dutch"*
**URL**: [Journal of Open Humanities Data](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.225)

- Training data: 15,154 lines (train), plus student annotations
- Multiple hands, mixed French/Dutch, 1876–1945
- Finetuned model CER: ~4–5% on in-domain test
- Note: the commonly cited 15,000-word threshold (Muehlberger et al., 2019) was *insufficient* for their multi-hand material

### D. CATMuS Medieval Benchmark (2024)

**URL**: [HuggingFace dataset](https://huggingface.co/datasets/CATMuS/medieval) / [ICDAR 2024 paper](https://inria.hal.science/hal-04453952/file/ICDAR24___CATMUS_Medieval-1.pdf)

- Dataset: 200+ manuscripts, 160,000+ lines, 5M+ characters, 10 languages, 8th–16th c.
- Kraken finetuning on CATMuS medieval base model: ~90% character accuracy achievable with ~1,000 lines (per GitHub issue #711)

### E. CREMMA Medieval — Kraken Models (2021)

**URL**: [GitHub: HTR-United/cremma-medieval](https://github.com/HTR-United/cremma-medieval)

- Training data: 17,431 lines from 10 Old French manuscripts (13th–14th c.)
- Kraken model accuracy: 89.19% (~10.8% CER) for the "Arabica" release
- Individual manuscript line counts: ranged from 153 to 6,148 lines

### F. GT4HistOCR — Printed Historical OCR (Springmann, Reul, Dipper, Baiter — 2018)

**Paper**: *"Ground Truth for training OCR engines on historical documents in German Fraktur and Early Modern Latin"*
**URL**: [arXiv:1809.05501](https://arxiv.org/abs/1809.05501) / [JLCL](https://jlcl.org/article/view/220)

- Dataset: 313,173 line pairs (printed, 15th–19th c.)
- Pretrained OCRopus models: 95% accuracy (early printings) to 98% (19th c. Fraktur)
- Mixed models on incunabula: average 95.4% accuracy (~4.6% CER) on unseen books
- Worst individual book: 91.9% accuracy (~8.1% CER)

### G. Kraken + ByT5 for Early Printed Books (2024)

**URL**: [IRIS UNINA](https://www.iris.unina.it/retrieve/370597be-56d8-4e83-9354-7a5973ef45ac/Modular%20Pipeline%20for%20Text%20Recognition%20in%20Early%20Printed%20Books%20Using%20Kraken%20and%20ByT5.pdf)

- Dataset: 10,000+ line pairs (early printed Latin, 15th c.)
- Kraken baseline CER: 19.9% → improved to 15.1% with ByT5 post-correction
- Smaller experiment: 2,000 lines → 5,000 lines for final model

---

## 3. Kraken GitHub Issues & Maintainer Guidance

### Issue #711 — The Definitive Maintainer Response

**URL**: [github.com/mittagessen/kraken/issues/711](https://github.com/mittagessen/kraken/issues/711)

Benjamin Kiessling (Kraken maintainer) provided the most detailed public guidance on dataset sizing:

> *"A 1000 line dataset isn't ideal... but there isn't a fundamental reason why it shouldn't train to at least **70-80% character accuracy**."*

He demonstrated: **1,000 lines → 74% CA (26% CER) from scratch after ~100 epochs**, and **~90% CA (10% CER) when finetuning from CATMuS medieval**.

> *"Let's say writing in your script or the document style you want to recognize is very regular, has a small glyph inventory, not much orthographic variation, and you don't require exceptionally high character accuracy (<1%), you'll be able to get away with **a couple of thousand well-sampled lines**."*

> *"If you want a model that recognizes a wide-range of handwriting styles and/or print in a script with lots of different glyphs and you use a very diplomatic transcription you'll require substantially more data."*

> *"Data quality is also important. People often create datasets by transcribing a whole document manually which will limit the generalization of the model in many cases. It is better to create a dataset of **100 pages from 1 page taken from 100 different documents** each than by transcribing 100 pages from a single one."*

### Skidzun/kraken_training — Community Rule of Thumb

**URL**: [github.com/Skidzun/kraken_training](https://github.com/Skidzun/kraken_training)

> *"There are no clear rules when to apply which approach but generally speaking a base model is a valid option until you have at least **10,000 words of training data for each scribal hand**."*

---

## 4. Comparison Points — Adjacent HTR Systems

### Transkribus (PyLaia / HTR+)

[Transkribus Help Center](https://help.transkribus.org/data-preparation):

> *"Between 5,000 and 15,000 words (around **25–75 pages**) of transcribed material are required to start."*
> *"For handwritten documents, our advice is to train the model on at least **10,000 words per hand**."*
> *"**25–50 transcribed pages** are enough to start training."* → 2–5% CER achievable.

**Greek manuscripts (John Chrysostom)** — [HAL: hal-03880102](https://hal.science/hal-03880102v4/document): models with ~1,000 words performed below the 20% CER threshold; a general model with **25,621 words achieved 4.60% CER**.

**Pracalit script (Nepalese manuscripts)** — [johd.90](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.90): 441 training pages + 242 validation pages; CER under 10% considered "effective" (Muehlberger et al., 2019).

**Tibetan cursive (Drutsa)** — [TibSchol HTR tools](https://www.oeaw.ac.at/fileadmin/Institute/IKGA/project_tibschol/PDFs/Online_documents/TibSchol_HTR_tools_v2_1.pdf): 422 training folios, 44 validation → **CER 1.40%**.

### Calamari

**Paper**: Wick, Reul, Puppe (2020). *"Calamari - A High-Performance Tensorflow-based Deep Learning Package for Optical Character Recognition"*
**URL**: [DHQ](https://dhq-static.digitalhumanities.org/pdf/000451.pdf) / [arXiv:1807.02004](https://arxiv.org/pdf/1807.02004)

- From scratch with 50 lines on historical prints: ~10% CER
- With pretraining + 50 lines: CER drops to ~2–5%
- 60–100 lines with data augmentation: CER below 2% (printed, pretrained)
- Caveat: these are *printed* with pretrained base models. From-scratch manuscript HTR is much harder.

**Reul et al. (2021)** — [arXiv:2106.07881](https://arxiv.org/pdf/2106.07881): from-scratch finetuning on Camerarius CER ~2.98%; with LSH-4 base model finetuning CER ~1.50% — using a general model as starting point cuts CER by half.

### HTR-United

**Paper**: Chagué, Clérice (2023). *"HTR-United, a solution towards a common for HTR training data"*
**URL**: [HAL: hal-04094233](https://inria.hal.science/hal-04094233v1/document)

- Catalog as of 2023: 78 datasets, 1M+ lines, 44M+ characters, 21 languages
- Core philosophy: shared datasets enable smaller projects to **fine-tune rather than train from scratch**, dramatically reducing data requirements
- No minimum-line-count prescription; ecosystem built on the premise that from-scratch training is prohibitively data-hungry for most projects

---

## 5. Realistic CER Expectations for From-Scratch Training

### What is "good" CER in historical HTR?

| CER Range | Quality | Practical Utility |
|-----------|---------|-------------------|
| <2% | Excellent | Near-human; minimal correction |
| 2–5% | Very good | Light correction; scholarly edition prep |
| **5–10%** | **Good/usable** | **Searchable; manageable correction; Muehlberger 2019 "effective" threshold** |
| 10–20% | Marginal | Keyword search; heavy correction |
| 20–30% | Poor | Faster to transcribe manually |
| >30% | Unusable | Manual transcription faster |

### From-scratch CER expectations by dataset size (synthesized)

| Lines | Expected CER (from scratch, single hand, small script) | Source |
|-------|-------------------------------------------------------|--------|
| 800 | 25–35% | Kraken docs lower bound for printed Arabic; worse for manuscript |
| 1,000 | 20–30% | GitHub issue #711: 74% CA (26% CER) demonstrated |
| 2,000 | 15–25% | Interpolated from multiple sources |
| 3,000–5,000 | 10–20% | Transkribus Greek experiment: ~1,000 words still <20% CER |
| 5,000–10,000 | 5–15% | CREMMA: 17k lines → 10.8% CER; ARletta: 15k lines → ~5% CER (finetuned) |
| 10,000–15,000 | 5–10% | Reul 2022: 12.5k lines → 6.22% CER (mixed, not from scratch) |
| 15,000–50,000 | 2–5% | TRIDIS: 245k lines → 4.8% CER; Transkribus Greek: 25k words → 4.6% CER |
| 50,000+ | <2% | GT4HistOCR: 313k lines → 1.5–5% CER; TibSchol: 422 folios → 1.4% CER |

---

## 6. Sogdian-Specific and Adjacent Script Work

### Avestan OCR with Kraken (closest analogue)

**Repository**: [`Nikyek/avestan-ocr-kraken-v1`](https://huggingface.co/Nikyek/avestan-ocr-kraken-v1)
**Pipeline**: [Avestan-Text-Processing-Suite](https://github.com/niktayek/Avestan-Text-Processing-Suite)

- Avestan is the closest script relative to Sogdian — both Middle Iranian, derived from Imperial Aramaic, RTL
- Project trained manuscript-specific Kraken recognition models (0040, 0088, 0089, 0090, 0091, 0093, 4000, 4210, TD2, lb2)
- Multiple segmentation models also trained
- Project explicitly notes: *"Others working on similar scripts (Pahlavi, Sogdian) could fine-tune these"*
- No published CER numbers, but working Kraken models for Avestan manuscripts prove feasibility
- Mirrored variants trained for Persian manuscripts (RTL handling)
- **Downloaded to `models/kraken/avestan_ms0040.mlmodel`** (~16 MB) — primary Avestan base for Sogdian fine-tuning
- **Also downloaded `models/kraken/sephardi_hebrew_rtl.mlmodel`** (~16 MB) — Hebrew RTL abjad base, secondary option

### Old Uyghur OCR (script descendant of Sogdian)

**Paper**: *"Old Uyghur OCR: The First Work-in-Progress via Reproducing Fine-tuning of VLMs"* (2025)
**URL**: [Academia.edu](https://www.academia.edu/143489426/Old_Uyghur_OCR_The_First_Work_in_Progress_via_Reproducing_Fine-tuning_of_VLMs)

- Training data: 525 manually annotated pages (from Altun Yaruk Sudur)
- Architecture: LLaMA-3.2-11B-Vision (not Kraken, but script-family data point)
- Result: **CER 5.46%**, NED 0.286 on held-out test
- Caveats: single woodblock style, small training data, binarized scans
- Relevance: Old Uyghur script is a direct descendant of Sogdian cursive — ~500 pages can yield ~5% CER with modern architectures (finetuning a VLM, not from-scratch CRNN)

### Arabic-Script Low-Resource HTR (OpenITI)

**Paper**: *"OpenITI MAKHZAN"* (2026) — [johd.465](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.465)

- Pre-2017 open-source Arabic-script OCR: 60–75% character accuracy
- Kraken optimized for Arabic-script achieved state-of-the-art
- **Lacuna reconstruction pretraining** (Vogler et al., 2022, [ACL 2022](https://aclanthology.org/2022.findings-naacl.15.pdf)): self-supervised pretraining on hundreds of thousands of unlabeled lines, then finetune on tens of labeled lines — significant CER reduction
- Relevance: for RTL connected scripts with small glyph inventories, Kraken is the best open-source option; self-supervised pretraining on unlabeled data can dramatically reduce labeled data needs

### Dead Sea Scrolls (Hebrew, RTL) with Kraken

**Paper**: *"Improving OCR for Historical Texts of Multiple Languages"* (2025) — [arXiv:2508.10356](https://arxiv.org/html/2508.10356v1)

- Kraken achieved **96.9% mean character accuracy** on Hebrew Dead Sea Scroll fragments
- Kraken explicitly noted as capable of processing RTL languages
- Data augmentation critical for the small fragment dataset

### Sogdian Unicode Encoding

**Document**: [Unicode proposal N4815](https://unicode.org/wg2/docs/n4815-sogdian.pdf)

- Sogdian script: **42 characters** (21 letters, 1 phonogram, 11 diacritics, 4 numbers, 5 punctuation)
- Conjoining abjad (like Arabic) — letters connect and change shape based on position
- RTL horizontal, also written vertically
- "Formal" and "cursive" styles — significant glyph variation
- Used over 9 centuries — substantial diachronic variation

**Implication**: positional forms (isolated, initial, medial, final) effectively multiply the glyph inventory. While the Unicode block has 42 characters, the visual forms a Kraken model must learn could be 80–120+. This pushes the minimum line count higher than the 800-line figure for simple printed Arabic.

---

## 7. Caveats Specific to From-Scratch Training and RTL/Low-Resource Scripts

### A. No base model = no transfer learning advantage

Every paper and the Kraken maintainer emphasize that **finetuning from even an out-of-domain pretrained model dramatically outperforms training from scratch**. Reul et al. (2022) showed 85% CER reduction with just 2 pages when using a pretrained model vs from scratch. For Sogdian, no existing Kraken base model — this is the hardest starting position.

**Mitigation**: finetune from the Avestan Kraken models (downloaded to `models/kraken/`). Even an out-of-domain model from a related script family (Syriac, Arabic, Hebrew) could help. Maintainer: *"even starting from an out-of-domain MM yields large improvements (>50%) over starting from scratch."*

### B. RTL and conjoining script challenges

- Kraken supports RTL natively (explicitly a feature: "Right-to-Left, BiDi, and Top-to-Bottom script support")
- Baseline detection for RTL scripts may need special attention — use `horizontal-rl` reading order
- Conjoining forms mean the effective visual glyph inventory is 2–4× the Unicode character count. Each positional form must appear enough times in training data
- Diacritics (11 in Sogdian) are small visual features easily missed — need sufficient representation

### C. The "10-occurrence minimum" rule

Kraken documentation: characters occurring **fewer than 10 times** in the training set will likely not be recognized. For Sogdian with ~42 base characters × positional forms, ensure every variant appears 10+ times. This alone may require 1,000+ lines.

### D. Data diversity beats data volume

Maintainer's key advice: **100 pages from 100 different documents** >> 100 pages from one document. For Sogdian manuscripts, if only a few manuscripts are available, the model will overfit to those specific hands and fail on new material.

### E. Transcription quality is paramount

> *"If your training data has errors, such as typos in your transcription, kraken will learn those errors, so high-quality ground truth is the most critical factor for good results."* — Kraken docs

For a low-resource language like Sogdian, where expert transcribers are scarce, this is a major bottleneck. Every hour spent improving transcription accuracy is likely more valuable than an hour spent adding more lines.

### F. Hyperparameter sensitivity for from-scratch training

- `--min-epochs` set higher (20–50) to prevent premature early stopping
- Lower learning rates (`-r 0.0001` for large handwritten datasets)
- `--augment` is critical for small datasets
- Multiple trials with different hyperparameters may be needed

### G. The iterative bootstrapping approach

Standard workflow for low-resource scripts:
1. Transcribe ~1,000–2,000 lines manually
2. Train a rough model (expect 20–30% CER)
3. Use that model to pre-transcribe more pages
4. Correct the output (3–5× speedup per Transkribus)
5. Retrain with the expanded dataset
6. Repeat until CER is acceptable

---

## 8. Practical Recommendations for the Sogdian Case

### Immediate strategy

1. **Start with 1,000–2,000 lines** as an initial feasibility test. Expect 20–30% CER. Validates that Kraken can learn Sogdian at all.
2. **Finetune from the Avestan Kraken model** (`models/kraken/avestan_ms0040.mlmodel`) before committing to pure from-scratch. Even if the script doesn't match perfectly, the shared Aramaic-derived structure may provide useful feature detectors. Use `--resize add` to handle the different character set.
3. **Aim for 5,000 lines as the first serious milestone.** At this scale, expect 10–15% CER — usable for search and with manageable correction effort.
4. **Target 10,000+ lines for sub-10% CER.** This is the threshold where the model becomes genuinely productive.
5. **Prioritize diversity**: if multiple Sogdian manuscripts are accessible, take 50–100 lines from each rather than 5,000 from one.

### Data preparation specifics for Sogdian

- Use Unicode Sogdian block (U+10F00–U+10F2F) for transcriptions
- Decide on transcription conventions early (diacritics? normalize positional forms? ligatures?) — consistency matters more than the specific choice
- Ensure every Sogdian character appears 10+ times in the training set — audit character frequencies before training
- Use eScriptorium for annotation — integrates directly with Kraken and supports RTL scripts
- Export as PAGE XML or ALTO XML for `ketos train`

### Training configuration (Kraken's recommended for large handwritten)

```bash
ketos --workers 4 -d cuda train \
  --augment \
  -f binary \
  --min-epochs 30 \
  -w 0 \
  -r 0.0001 \
  -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]' \
  dataset.arrow
```

`--min-epochs 30` prevents early stopping from killing training before the model produces sensible output (critical for from-scratch). Lower LR for large handwritten datasets. `--augment` essential for small datasets.

### Realistic timeline

| Phase | Lines | Effort | Expected CER |
|-------|-------|--------|--------------|
| Feasibility test | 1,000 | 1–2 weeks transcription | 20–30% |
| First usable model | 3,000–5,000 | 1–2 months (with bootstrapping) | 10–15% |
| Production model | 10,000+ | 3–6 months | 5–10% |
| Multi-manuscript general model | 20,000+ | 6–12 months | 2–5% |

---

## 9. Key Citations

1. **Kraken official documentation**: [kraken.re](https://kraken.re/main/) — training tutorial, recognition training, introduction to ATR
2. **Kiessling, B.** (Kraken maintainer). GitHub Issue #711. [github.com/mittagessen/kraken/issues/711](https://github.com/mittagessen/kraken/issues/711)
3. **Reul, C., Springmann, U., Wick, C., Puppe, F.** (2018). "Improving OCR Accuracy on Early Printed Books by Combining Pretraining, Voting, and Active Learning." *JLCL* 33(1). [DOI: 10.21248/jlcl.33.2018.220](https://jlcl.org/article/view/220)
4. **Springmann, U., Reul, C., Dipper, S., Baiter, J.** (2018). "Ground Truth for training OCR engines on historical documents in German Fraktur and Early Modern Latin." *JLCL* 33(1), 97–114. [arXiv:1809.05501](https://arxiv.org/abs/1809.05501)
5. **Reul, C., Tomasek, S., Langhanki, F., Springmann, U.** (2022). "Open Source Handwritten Text Recognition on Medieval Manuscripts using Mixed Models and Document-Specific Finetuning." [arXiv:2201.07661](https://arxiv.org/abs/2201.07661)
6. **Torres Aguilar, S., Jolivet, V.** (2023). "Handwritten Text Recognition for Documentary Medieval Manuscripts." [HAL: hal-03892163](https://hal.science/hal-03892163)
7. **Torres Aguilar, S.** (2024). "TRIDIS v2." [Zenodo: 10.5281/zenodo.13862096](https://zenodo.org/records/13862096)
8. **Wick, C., Reul, C., Puppe, F.** (2020). "Calamari - A High-Performance Tensorflow-based Deep Learning Package for Optical Character Recognition." *DHQ* 14(1). [DHQ](https://dhq-static.digitalhumanities.org/pdf/000451.pdf)
9. **Muehlberger, G. et al.** (2019). "Transforming scholarship in the archives through handwritten text recognition." *Journal of Documentation* 75(5). (Establishes <10% CER threshold for effective automatic transcription, and the 15,000-word guideline.)
10. **Chagué, A., Clérice, T.** (2023). "HTR-United." [HAL: hal-04094233](https://inria.hal.science/hal-04094233v1/document)
11. **Nikyek** (2024). "Avestan OCR Training and Application – Kraken + eScriptorium." [HuggingFace: avestan-ocr-kraken-v1](https://huggingface.co/Nikyek/avestan-ocr-kraken-v1)
12. **Old Uyghur OCR** (2025). [Academia.edu](https://www.academia.edu/143489426/Old_Uyghur_OCR_The_First_Work_in_Progress_via_Reproducing_Fine-tuning_of_VLMs) — CER 5.46% with 525 pages
13. **Vogler, N. et al.** (2022). "Lacuna Reconstruction: Self-Supervised Pre-Training for Low-Resource Historical Document Transcription." [ACL 2022](https://aclanthology.org/2022.findings-naacl.15.pdf)
14. **Transkribus documentation**: [help.transkribus.org](https://help.transkribus.org/data-preparation)
15. **Unicode Sogdian proposal**: [N4815](https://unicode.org/wg2/docs/n4815-sogdian.pdf)

---

## Bottom Line

**For training a Kraken HTR model from scratch for Sogdian manuscripts:**

- **Absolute minimum**: ~1,000 lines (expect ~25% CER — barely usable)
- **Realistic first target**: 3,000–5,000 lines (expect 10–15% CER — searchable, correctable)
- **Production quality**: 10,000+ lines (expect 5–10% CER — good enough for scholarly work)
- **The single highest-impact action**: finetune from the existing Avestan Kraken model (`models/kraken/avestan_ms0040.mlmodel`) before committing to pure from-scratch. Even a loosely related base model can cut data requirements by 50% or more.

The Sogdian script's small glyph inventory (~42 base characters) works in your favor, but the conjoining/positional forms, RTL direction, diacritics, and manuscript-only domain all push requirements higher than the 800-line printed-Arabic baseline. Plan for 3,000–5,000 lines as the point where the model becomes genuinely useful, and budget for iterative bootstrapping cycles to get there efficiently.