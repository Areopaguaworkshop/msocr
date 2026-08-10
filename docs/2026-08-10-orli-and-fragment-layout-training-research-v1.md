# Orli, Fragment-Aware Layout Training, and Kraken Maintainer Contact — v1

**Date:** 2026-08-10
**Version:** v1
**Status:** Research brief supporting the Christian Sogdian fragment HTR plan
**Scope:** Kraken's orli successor to BLLA, layout-training approaches for fragmented manuscripts, and contact for Benjamin Kiessling (Kraken maintainer)

Companion to:
- [Christian Sogdian Fragment HTR and CER-Reduction Plan](./2026-08-02-christian-sogdian-fragment-htr-plan-v1.md)
- [Kraken Fragmented-Manuscript Limitations and C2 Image-Source Report](./2026-08-02-kraken-fragmented-manuscript-limitations-v1.md)

---

## 1. Orli — what it is and what it replaces

Orli = **Ordered Regression of Lines**. It is Benjamin Kiessling's successor to
BLLA. It is no longer hypothetical: paper released June 2026, base model
published June 5 2026, code public on GitHub.

| Pointer | URL |
|---|---|
| Paper | <https://arxiv.org/abs/2606.04166> |
| Base model | <https://zenodo.org/record/20558179> (DOI 10.5281/zenodo.20558179) |
| Code | <https://github.com/mittagessen/orli> |
| License | Apache 2.0 |

### 1.1 What problem it solves that BLLA does not

BLLA is a two-stage pipeline: pixel-wise heatmap → polygonize → **separate
hand-coded geometric reading-order heuristic**. That heuristic "struggles with
marginalia, multiple columns, tables, and source-specific editorial
conventions" (orli paper, Abstract). Orli makes reading order a native,
trainable part of the prediction.

### 1.2 Architecture

| Component | Detail |
|---|---|
| Vision encoder | ConvNeXtV2-tiny |
| Multi-scale adapter | RT-DETR-style hybrid encoder (top-down FPN + bottom-up PAN), projects to 256-dim shared channel |
| Decoder | LLaMA-style transformer, 12 layers, embedding dim 576, intermediate dim 1536, GQA (9 query heads / 3 KV heads), RoPE, causal self-attention + unmasked cross-attention to encoder memory |
| Baseline representation | **Chord-frame regression**: anchor point + orientation + extent + perpendicular offsets; 4 refinement iterations from decoder layers L2→L5→L8→L12, plus a local visual refiner that samples encoder features along the predicted baseline |
| Input | 1920×1440 (high-resolution variant) |
| Precision | **bfloat16 only** — README warns "other precisions are likely to cause runaway generation" |

The chord-frame parameterization is the key innovation over direct point
regression (jittery polylines) and Bézier control points (not co-located with
the curve).

### 1.3 What it predicts

- Text-line **baselines** as curves (chord-frame polylines), emitted
  **autoregressively in reading order**.
- Currently **lacks**: line-type classification and semantic block markers
  (explicitly noted as missing in the Zenodo model card).

### 1.4 Training data and assumptions

- ~200k pages across **ten writing systems** (Zenodo; paper says 196,691).
- Compiled from PageXML/ALTO into Arrow datasets (NOT compatible with Kraken's
  compiled datasets — separate `orli compile` toolchain).
- Uses implicit reading order (sequence of `<line>` elements in the source
  file).
- **No mention of fragments, partial pages, damaged inputs, or non-page
  images** anywhere in the paper, README, Zenodo card, or model card.

The Zenodo card says the model "should perform reasonably well on most printed
and handwritten historical texts in non-CJK writing systems" — i.e. this is a
**full-page model**. The same wording family as the BLLA card's "non-fragmentary"
caveat.

### 1.5 Release status

Released and usable now. Timeline:

- 2026-06-02: paper on arXiv
- 2026-06-05: base model on Zenodo
- 103 commits on GitHub as of 2026-08-10, Apache 2.0

Usage:

```bash
kraken get 10.5281/zenodo.20558179
kraken --precision bf16-mixed -i input.jpg output.xml -x segment -bl \
  --model orli_base.safetensors
```

Programmatic:

```python
from PIL import Image
from orli.pred import segment
im = Image.open("input.jpg")
segmentation = segment(im, "/path/to/orli_base.safetensors")
```

Fine-tuning is supported:

```bash
orli --config finetune.yaml train --load "$MODEL"
```

### 1.6 Kraken integration

Via the Kraken 7.0 plugin system (entry-point based; same pattern as
[mittagessen/dfine_kraken](https://github.com/mittagessen/dfine_kraken)). The
orli README states explicitly: "Orli integrates with kraken 7 through its model
plugin system."

Kraken 7.0 release notes: "A plugin system now allows easy extension of kraken
functionality with new segmentation, recognition, and reading order
implementations. […] Plugins are distributed as regular Python packages. After
installation, kraken discovers them automatically through entry points. Plugin
model files are then used exactly like native kraken model files."

The legacy `blla.segment()` API is **deprecated in 7.0, removed in 8.0**.

### 1.7 Reported scores (for context)

| Benchmark | Precision | Recall | F1 |
|---|---|---|---|
| Orli test set | 0.9554 | 0.9564 | 0.9559 |
| cBAD 2019 (base) | 0.9378 | 0.9302 | 0.9340 |
| cBAD 2019 (fine-tuned) | 0.9395 | 0.9306 | 0.9351 |

Reading-order: near-perfect coverage on OHG and FCR; ABP improves dramatically
after fine-tuning (Kendall tau: 0.2878 → 0.8972).

### 1.8 Orli on fragments — honest assessment

**No evidence orli was trained or tested on fragmentary inputs.** Its
autoregressive "emit one line at a time" design is conceptually more robust to
sparse inputs than a dense heatmap (it does not need to explain every pixel),
but:

- The base model was trained on full pages; the decoder expects page-scale
  context.
- A small fragment with few lines is out-of-distribution.
- The reading-order head has no obvious fallback for "single line, no
  neighbors."
- bf16-only inference is a hard constraint (runaway generation on other
  precisions).

The path to use orli on fragments is **fine-tuning on annotated fragment data**
(via `orli --config finetune.yaml train --load "$MODEL"`). The paper shows
strong out-of-domain fine-tuning results, so this is plausible but
**untested territory for fragmentary manuscripts**.

---

## 2. Layout-training landscape for fragmented / damaged manuscripts

There is **no published deep-learning solution for line segmentation on
manuscript fragments**. What exists, ranked by fragment-fit:

### 2.1 Object detection (YOLO-OBB / D-FINE) — best conceptual fit

Object detectors treat each line as an independent instance rather than
segmenting a continuous page. This is inherently more robust to
sparse/fragmented inputs — the model does not need to understand the full-page
layout, just "is there a line here?"

| Work | Evidence |
|---|---|
| **YALTAi** (Clérice, 2022) | YOLOv5 injected into Kraken 4.1's segmentation pipeline. "YOLOv5 severely outperforms Kraken on the same dataset for region segmentation, both in terms of evaluation metrics and GPU efficiency" — especially on small datasets (≤1110 samples). [JDMDH paper](https://jdmdh.episciences.org/12747) |
| **D-FINE kraken** (Kiessling, 2025) | Transformer-based object detector for document layout; trains lines and regions jointly; already integrated as a Kraken 7.0 plugin with pretrained models on the LADaS dataset (37 SegmOnto region types). [github.com/mittagessen/dfine_kraken](https://github.com/mittagessen/dfine_kraken) |
| **YOLOv11-OBB** (Torres Aguilar, 2025) | Benchmarked 5 architectures on medieval manuscripts (e-NDP, CATMuS, HORAE). "Using Oriented Bounding Boxes (OBB) is not a minor refinement but a fundamental requirement for accurately modeling the non-Cartesian nature of historical manuscripts." YOLOv11x-OBB significantly outperformed Transformer-based models (Co-DETR, Grounding DINO). [arXiv:2506.20326](https://arxiv.org/abs/2506.20326) |
| **YOLO for Ottoman manuscripts** | 91% F1 for text-line segmentation using YOLO instance segmentation with oriented bounding boxes. [Sabanci University](https://research.sabanciuniv.edu/id/eprint/52563/) |
| **YOLOv5-OBB for Greek polytonic** | +7.5% mAP over rectangular boxes using Circular Smooth Label (CSL) angle encoding. [ICDAR 2023 workshop](https://users.iit.demokritos.gr/~bgat/icdar%202023_workshops_pp213_225_LNCS%2014194.pdf) |

**Caveat:** none of these tested on fragment inputs specifically. They detect
**bboxes**, not baselines — for Kraken recognition you'd need to convert bboxes
to baseline polylines or use Kraken's bbox recognition mode (supported since
7.0.3 via `--linetype bbox`).

### 2.2 Surya

- Text-line detection: small EfficientViT-based segformer, trained from scratch
  on document line annotations. Sub-second latency.
- Main OCR model: 650M-param VLM (Qwen3.5-style) handling layout + recognition
  + tables.
- **Limitation**: "It is for printed text, not handwriting (though it may work
  on some handwriting)." — [GitHub README](https://github.com/datalab-to/surya)
- No benchmarks vs Kraken BLLA found. No mention of fragments. Outputs
  axis-aligned bboxes + 4-point polygons per line, not baselines.

### 2.3 eScriptorium / HTR-United

- eScriptorium uses Kraken as its backend. Workflow: segment with an existing
  model → manually correct → fine-tune → repeat. Assumes full-page inputs.
- HTR-United is a data catalog, not a model catalog. **No Sogdian datasets
  found.**
- CREMMA Medieval used SegmOnto ontology for medieval manuscript layout.
- **No documented approaches or model variants for fragmented manuscripts** in
  either ecosystem.

### 2.4 Custom `ketos segtrain` on fragment-only data

**No documented cases found.** Closest evidence:

- Kraken issue [#656](https://github.com/mittagessen/kraken/issues/656):
  Syriac fine-tuning on full pages.
- Kraken issue [#764](https://github.com/mittagessen/kraken/issues/764):
  InterlinearLine fine-tuning on full pages.
- Digital Orientalist tutorial (2023): recommends 30–50 pages for fine-tuning,
  full-page book-specific tasks.
- BLLA base model card explicitly says "non-fragmentary handwritten and
  machine-printed document pages" — [Zenodo 14602569](https://zenodo.org/records/14602569).

**Counter-evidence (Dead Sea Scrolls):** the DSS project (Dershowitz et al.)
used Kraken BLLA on highly fragmentary DSS material and reported "Layout
analysis is independent of binarization and works very well even on highly
fragmentary and damaged material (such as the Dead Sea Scrolls and the
Genizah)." [DSS paper PDF](https://www.cs.tau.ac.il/~nachumd/papers/DSS.pdf)

**Caveat:** their "fragments" averaged 5–6 lines and ~75 characters each —
substantially larger than the E27 pilot fragments in msocr (which produced
**zero** usable lines from BLLA). The DSS result is real but does not
generalize down to our smallest fragments.

### 2.5 Classical CV alternatives

| Work | What it does |
|---|---|
| **DSS fragment isolation** (Brown-deVost et al., 2024) | Faster R-CNN for calibration bar detection → recto/verso alignment → binarization → morphological closing → intersection of recto/verso masks. 0.97 IoU for fragment segmentation. This is fragment **isolation**, not line segmentation. [arXiv:2406.15692](https://arxiv.org/abs/2406.15692) |
| **MTEM for DSS ink/parchment** (2026) | Multispectral thresholding + energy minimization. Segments ink, parchment, background, holes, and rice paper as separate regions. Classical CV, not DL. [IJDAR](https://link.springer.com/article/10.1007/s10032-025-00564-4) |
| **Ostracon analysis** (Faigenbaum et al.) | Connected components + Hough transform for line detection on very small fragments. Classical CV. |
| **msocr `row_bands`** (this project) | k-means Y-clustering of ink components with human-supplied `expected_lines` count. Works on isolated fragment images by design. Deterministic. Already in production. |

### 2.6 Damage-aware / multi-task segmentation

**No model found that explicitly predicts gaps/damage as a separate output
class.** The closest is the DSS MTEM method (segments holes/rice paper as
regions, classical CV, not a learned predictor). msocr's `evaluate-layout`
already *measures* gap localization as an evaluation metric — ahead of what
predictors do.

### 2.7 Bottom line for msocr

| Approach | Fragment-fit | Evidence | Risk |
|---|---|---|---|
| **Orli fine-tuned on fragments** | Speculative, high potential | Autoregressive generation may handle sparse lines better than a dense heatmap; Kraken 7 plugin makes integration trivial; paper shows strong OOD fine-tuning | Base model is full-page; no fragment tests published; bf16-only constraint |
| **YOLO-OBB / D-FINE line detection** | Best conceptual fit | Object detection is inherently sparse-input friendly; YALTAi showed YOLO beats Kraken on small datasets; D-FINE already a Kraken 7 plugin; YOLOv11-OBB is SOTA for complex historical manuscripts | Outputs bboxes, not baselines; bridgeable via Kraken 7.0.3 `--linetype bbox` or bbox→baseline conversion |
| **Current `row_bands` pipeline** | Proven, in production | Deterministic; no training; DSS project confirms classical-CV + Kraken recognition is viable on fragmentary material | Needs human `expected_lines` count; could be automated with a small regressor |
| **`ketos segtrain` on fragment-only data** | No documented cases | None | Full-page U-Net is the wrong inductive bias; high data requirement |

---

## 3. Kraken maintainer contact — Benjamin Kiessling

Dual affiliation: **Inria Paris (ALMAnaCH team)** + **EPHE / Université PSL
(AOROC lab, CNRS/ENS/EPHE)**. ORCID: [0000-0001-9543-7827](https://orcid.org/0000-0001-9543-7827).

### 3.1 Verified emails (all HIGH confidence)

| Email | Source |
|---|---|
| **`mittagessen@l.unchti.me`** | His personal domain; listed as author email in [kraken `pyproject.toml`](https://github.com/mittagessen/kraken/blob/main/pyproject.toml), PyPI, and the Kraken docs. The canonical open-source contact. |
| `benjamin.kiessling@psl.eu` | His own CV, April 2023 ([PDF on archeo.ens.fr](https://www.archeo.ens.fr/IMG/pdf/kiessling_cv_avril2023.pdf)). Institutional. |
| `Benjamin.kiessling@ens.psl.eu` | AOROC lab directory ([lab page](https://www.archeo.ens.fr/Kiessling-Benjamin.html?lang=fr)). Same inbox as `psl.eu`. |

**Recommended send order:** `mittagessen@l.unchti.me` first (the address he
chose to publish on his own software), then `benjamin.kiessling@psl.eu` as
fallback.

### 3.2 Preferred channel per Kraken docs

From [kraken.re](https://kraken.re/main/): "kraken is an open-source project
driven by community contributions. We warmly welcome feedback, pull requests,
**bug reports, and feature suggestions on our github repository**."

So GitHub Issues on `mittagessen/kraken` is the documented preferred channel.
Gitter `escripta/escriptorium` exists for broader community. No Discord /
Matrix / mailing list. GitHub Discussions is not enabled on the repo.

### 3.3 Suggested email draft

> Subject: orli on fragmentary manuscripts — has it been tested?
>
> Hi Benjamin,
>
> I work on HTR for fragmented Christian Sogdian manuscripts from the Turfan
> collection (Sims-Williams 1985 C2AV Berlin plates — Sogdian language in East
> Syriac script, Unicode U+0710). We use Kraken for recognition; for layout we
> currently bypass BLLA entirely with a classical CV row-band pipeline (ink-CC
> Y-clustering with a human-supplied line count), because the default BLLA
> model produces zero usable line crops on our fragments — we verified this in
> an E27 pilot.
>
> Two questions:
>
> 1. Has orli been evaluated on fragmentary / non-page inputs (small fragments,
>    sparse lines, partial pages)? The base model card and paper describe
>    full-page training; I haven't found fragment benchmarks.
> 2. If not, do you have a view on whether orli's autoregressive baseline
>    regression is a better starting point than a bbox object detector
>    (D-FINE/YOLO-OBB) for fragment line detection, before we invest in
>    fine-tuning either?
>
> Context: we have annotated PAGE XML with atomic per-fragment baselines and
> can fine-tune. I'd rather ask before committing a direction.
>
> Thanks — and thank you for Kraken.
> [your name / project]

The question is concrete and specific enough to answer in two sentences, and
it signals you've already done the homework (E27 pilot, row_bands workaround,
orli + D-FINE both on the table).

---

## 4. Confidence and gaps

- **Orli architecture, release, integration:** HIGH — confirmed from the
  paper, Zenodo model card, and GitHub README.
- **Orli on fragments:** gap — no evidence either way; the question is open
  and worth asking the maintainer.
- **Object-detection approaches (YOLO-OBB, D-FINE):** HIGH for general
  manuscript line detection; MEDIUM-LOW for fragment-specific use (no
  fragment-specific benchmarks published).
- **DSS BLLA success:** HIGH — quoted from the DSS paper; the size caveat
  (their fragments average 5–6 lines) is the key limit on transferability to
  our smallest E27 fragments.
- **Maintainer emails:** HIGH — all three verified from primary sources
  (his CV, his lab page, kraken's own pyproject.toml and PyPI listing).
- **`ketos segtrain` on fragments:** gap — no documented cases; this is the
  frontier.