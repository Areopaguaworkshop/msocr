# Kraken HTR on Fragmentary Manuscripts — Engineering Reference

> How Kraken's baseline segmentation and recognizer handle folios where a single
> logical line is broken across inked fragments with blank/damaged lacunae in
> between, and how to annotate and train on such sources.
>
> Companion to [`line-segmentation-strategy.md`](./line-segmentation-strategy.md)
> (our per-row crop policy) and [`ANNOTATION.md`](./ANNOTATION.md) (our editor
> workflow). Research via @librarian, mid-2026.

## TL;DR

- Kraken's baseline segmenter (BLLA) is a U-Net that labels pixels as
  baseline/region/separator classes, then vectorizes heatmaps into baselines
  and computes polygons. The recognizer crops each polygon, masks outside it to
  black, piecewise-affine-rectifies to a strip, and feeds a CNN+RNN+CTC net.
- **The default BLLA model is explicitly trained on "non-fragmentary" pages**
  (Zenodo model card). Gaps, holes, and faded ink cause it to *split* one line
  into multiple baselines (kraken #745, #677). Fine-tuning on your own
  fragmentary Sogdian pages is the documented remedy.
- **Annotation convention: one baseline per contiguous inked fragment.** Do not
  draw a single baseline across a lacuna. The polygon of such a baseline would
  include blank pixels with no GT characters, which confuses both the seg
  target tensor and the CTC recognizer.
- **Uncertain glyphs:** use Leiden underdot (U+0323 COMBINING DOT BELOW, e.g.
  `ạ`). Completely lost stretches: split the line. A lacuna *token* in the GT
  string (e.g. `[...]`, `□`) is experimental and unvalidated in Kraken — the
  CTC recognizer has no semantic "nothing here" label distinct from its
  alignment-blank.
- **No per-pixel "don't-care" zone exists** in Kraken's seg or rec training.
  `--suppress-*`, `--valid-*`, `--bounding-regions`, and the page-level
  `segment(mask=...)` are the only exclusion knobs. `skip_empty_lines=True`
  (default) drops lines with empty GT but does *not* drop lines whose only GT
  is a lacuna token.
- **Coming:** CurT / orli (Bézier-curve line predictor) may handle fragmentary
  baselines better than BLLA. No release date. Track
  [github.com/mittagessen/orli](https://github.com/mittagessen/orli).

---

## 1. How Kraken represents and consumes a line

### 1.1 BLLA segmentation (trainable baseline segmenter)

Three stages (Kiessling, ICFHR 2020, [hal-04442992](https://hal.science/hal-04442992/document)):

1. **Multi-label pixel classification** — a ResNet-34 U-Net labels every pixel
   with one or more classes: baseline classes, region classes, and two
   auxiliary separator classes (line start/end) used to recover orientation.
2. **Baseline vectorization** — heatmap → Gaussian smooth → hysteresis
   threshold → skeletonize → endpoint extraction → polyline rectification.
3. **Polygonization** — a variable-width polygon is computed around each
   baseline from the distance of connected components to the labelled area.

Docs: [kraken.re/6.0.0/advanced/segmentation.html](https://kraken.re/6.0.0/advanced/segmentation.html).

### 1.2 Line container

Each line is a `BaselineLine` dataclass
([kraken/containers.py](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/containers.py)):

```python
@dataclass
class BaselineLine:
    id: str
    baseline: List[Tuple[int, int]]   # polyline
    boundary:   List[Tuple[int, int]] # closed polygon (first==last)
    text: Optional[str]               # transcription
    base_dir: Optional[str]           # 'L' or 'R'
    tags: Dict[str, str]              # {'type': 'default'|'Heading'|...}
    regions: List[str]                # associated region IDs
```

PageXML serializes this as `<TextLine>` → `<Baseline points="…"/>` +
`<Coords points="…"/>` + `<TextEquiv><Unicode>…</Unicode></TextEquiv>`.
ALTO: `<TextLine BASELINE="…">` + `<Shape><Polygon POINTS="…"/></Shape>` +
`<String CONTENT="…" WC="1.0"/>`.

### 1.3 How the recognizer sees the line

`extract_polygons()` in `kraken/lib/segmentation.py`:

1. Crops the polygon's bounding box from the page.
2. Masks pixels **outside** the polygon to 0 (black).
3. Piecewise-affine-warps the baseline+polygon into a straight horizontal
   strip (simple rotation for 2-point straight baselines).
4. Feeds the strip to a CNN+RNN trained with **CTC loss** — a segmentation-less
   sequence classifier mapping image frames → character sequence.

> **Implication:** the recognizer sees the *whole* polygon area. Anything
> blank inside the polygon becomes black input frames that CTC must align to
> *some* GT symbol (or to the CTC blank label). There is no "ignore" channel.

Refs: Kiessling DH abstract
([dh-abstracts.library.virginia.edu/works/9912](https://dh-abstracts.library.virginia.edu/works/9912));
`extract_polygons()` source.

---

## 2. Why fragmentary folios break the default model

### 2.1 The documented failure mode

kraken [#745 "Excessive horizontal segmentation"](https://github.com/mittagessen/kraken/issues/745):

> "Lines are segmented horizontally in regions that are in fact consistently
> left-right justified blocks of text… due to many factors that seem to
> interrupt the horizontal detection of the line, such as darkened
> parchment/paper, faded (esp. red) or erased ink, **holes in the manuscript
> material**, interlinear text and so on."

Maintainer (B. Kiessling): fine-tuning the base seg model on a few pages
"might get you better results" than adding post-processing; the longer-term
answer is the CurT/orli successor, not more BLLA knobs.

kraken [#677 "Broken/split lines in segmentation"](https://github.com/mittagessen/kraken/issues/677):
persistent split lines even after several training attempts; resolved by
fine-tuning on a domain segmentation model (`ubma_segmentation.mlmodel`).

The default BLLA model card on Zenodo
([zenodo.org/records/14602569](https://zenodo.org/records/14602569)) states it
outright:

> "The model performs reasonably well on most **non-fragmentary** handwritten
> and machine-printed document pages of moderate complexity."

It also notes the base "skews heavily towards Latin script." Sogdian is RTL
and out-of-domain — expect to fine-tune.

### 2.2 Polygon-offset sensitivity

kraken [#773](https://github.com/mittagessen/kraken/issues/773): changing the
polygonization offset from 8 px to 4 px caused a measurable CER regression on
dense historical MS. Polygon boundary quality directly drives recognition
accuracy — so polygons that bulge across blank lacunae are not free.

---

## 3. Annotation options for ink-vs-blank lines

| Option | What it means | Verdict |
|---|---|---|
| **(a) Split into separate `TextLine`s** | One baseline + polygon per inked fragment; no baseline across the gap. | ✅ **Recommended.** Matches eScriptorium workflow ("baseline is the central element; transcribe only what's inside the polygon"). |
| (b) One continuous baseline across the gap | Single polygon includes blank lacuna; recognizer sees black frames with no GT. | ❌ Causes seg splits (#745); CTC has no semantic "nothing" label; blank frames either emit spurious chars or suppress output. |
| (c) Continuous baseline + lacuna token in GT | e.g. `[...]` or `□` between the inked words. | ⚠️ Experimental. The model *will* learn the token if consistent, but Kraken has no validated convention; "Mind the Gap" (TrCroT, arXiv:2407.00250) failed to bracket lacunae reliably even with custom loss. |
| (d) Mask out the missing region | Page-level `segment(mask=…)` zeroes damaged pixels *before* segmentation. | ◻️ Inference-time only; stops the seg from *finding* lines in a zone. Doesn't help a line that legitimately *crosses* a gap. |

### 3.1 Within-line uncertainty (partial ink)

Use Leiden/epigraphic conventions, not free text:

| Siglum | Unicode | Use |
|---|---|---|
| Underdot | `U+0323` COMBINING DOT BELOW (`ạ`) | Partially legible glyph (ink faint but trace visible). |
| `…` / `[...]` | literal | Lost stretch of *known* length (one dot per char if countable). **Experimental in Kraken** — see Option (c). |
| `⟦abc⟧` | `U+27E6` / `U+27E7` | Scribal deletion/restoration (CATMuS adoption of Leiden). |
| `�` | `U+FFFD` | Our repo's current placeholder per `ANNOTATION.md` §7. Prefer underdot where the glyph is partly visible; reserve `�` for "there was a glyph here, I have no idea what." |

Refs: [Leiden Conventions](https://en.wikipedia.org/wiki/Leiden_Conventions);
[CATMuS guidelines](https://catmus-guidelines.github.io/html/guidelines/en/letters_numbers.html);
TEI `<gap>`/`<unclear>`/`<supplied>`/`<damage>`
([tei-c.org](https://www.tei-c.org/Vault/GL/P3/PH.htm));
[EpiDoc](https://epidoc.stoa.org/gl/dev/trans-lostcharapprox.html).

### 3.2 Kraken's official stance on transcription complexity

[kraken.re/7.0/introduction_to_atr.html](https://kraken.re/7.0/introduction_to_atr.html):

> "Very intricate transcription norms can be difficult for the model to
> reproduce. In general, it is often better to simplify your requirements to
> match the capabilities of the technology…"

And on diplomatic GT
([6.0 training tutorial](https://kraken.re/6.0.0/tutorials/training.html)):

> "Transcription has to be diplomatic, i.e. contain the exact character
> sequence in the line image, including original orthography."

The two combined imply: keep the alphabet small, consistent, and graphemic;
avoid editorial expansion sigla that the recognizer must invent without visual
evidence.

---

## 4. Recommended convention for fragmentary Sogdian folios

### 4.1 Rules

1. **One baseline per contiguous inked fragment.** Never draw a baseline
   across a physical gap (hole, tear, missing papyrus, blank margin).
2. **Reading order:** top-to-bottom; within a row, right-to-left (Sogdian is
   `horizontal-rl`). Each fragment of the same logical row is its own
   `TextLine`, ordered RTL in the PAGE XML's `ReadingOrder`.
3. **Diplomatic graphemic transcription** within each polygon. Normalize
   heterograms only via a documented project policy — and if you normalize,
   do it everywhere consistently.
4. **Uncertain glyphs:** `U+0323` underdot. Fully unknown but present glyph:
   `�`. Lost stretch of *known* length: split the line; do not encode as
   `[...]` in GT unless you are running a lacuna-token experiment.
5. **Lacunae as regions, not GT.** Mark physically damaged zones with
   `DamageZone` (see `ANNOTATION.md` §2) for layout/exclusion; do **not** put
   a baseline or transcription inside a `DamageZone`.
6. **Inference mask:** when running `kraken -i page.png … segment`, pass
   `-m mask.png` to suppress segmentation in known damaged bands.

### 4.2 PageXML — line crossing a lacuna (split approach, RTL)

```xml
<TextRegion id="r_main" custom="structure {type:MainZone;}">
  <Coords points="200,400 2200,400 2200,3200 200,3200"/>

  <!-- Right fragment (visual right = reading start for RTL Sogdian) -->
  <TextLine id="l5_r" custom="readingDirection {value:right-to-left;}">
    <Baseline points="1800,1200 1200,1205"/>
    <Coords points="1820,1180 1180,1185 1180,1230 1820,1225"/>
    <TextEquiv><Unicode>pr βγy mrtxmʾk</Unicode></TextEquiv>
  </TextLine>

  <!-- Left fragment: resumes after the lacuna, same original row -->
  <TextLine id="l5_l" custom="readingDirection {value:right-to-left;}">
    <Baseline points="800,1205 300,1200"/>
    <Coords points="820,1185 280,1180 280,1230 820,1235"/>
    <TextEquiv><Unicode>ZYn ʾkrtyh</Unicode></TextEquiv>
  </TextLine>

  <!-- Next row, intact -->
  <TextLine id="l6" custom="readingDirection {value:right-to-left;}">
    <Baseline points="1800,1280 300,1285"/>
    <Coords points="1820,1260 280,1265 280,1310 1820,1305"/>
    <TextEquiv><Unicode>rty δβrʾk ʾPZY βγy</Unicode></TextEquiv>
  </TextLine>
</TextRegion>
```

### 4.3 PageXML — partially legible inked line

```xml
<TextLine id="l8" custom="readingDirection {value:right-to-left;}">
  <Baseline points="1800,1440 300,1445"/>
  <Coords points="1820,1420 280,1425 280,1470 1820,1465"/>
  <!-- ạ = aleph with U+0323 underdot = faint/uncertain -->
  <TextEquiv><Unicode>rty ạḅγ ZY βγy</Unicode></TextEquiv>
</TextLine>
```

### 4.4 Relationship to our existing crop policy

[`line-segmentation-strategy.md`](./line-segmentation-strategy.md) already
prescribes **one padded line-band crop per manuscript row, preserving blank
gaps inside the row** for the *annotation* UI's auto-suggest step. This doc
refines what happens *after* annotation, when those rows are committed to
PAGE XML and fed to `ketos segtrain` / `ketos train`:

- The crop-stage "keep blanks inside the row band" rule is for the annotator's
  visual reference. At XML commit time, split each row band into per-fragment
  `TextLine`s following §4.2 — i.e. the *baseline* never crosses the gap,
  even though the *crop* did.
- `DamageZone` polygons (per `ANNOTATION.md`) should cover the gap region so
  the seg trainer can be told (via `--suppress-regions`) not to look there.

---

## 5. Training-side implications

### 5.1 What hurts

- **Seg model trained on baselines crossing blank zones** learns that baseline
  pixels can exist where there is no ink — directly contradicting the
  semantic-segmentation target. Symptom: more split lines, not fewer.
- **Rec model trained on polygons enclosing blank lacunae** gets black frames
  with no matching GT char. `skip_empty_lines=True` (default,
  `kraken/lib/dataset/recognition.py`) drops fully-empty GT lines, but does
  **not** drop a line whose GT is e.g. `"pr βγy […] mrtxmʾk"` — the model
  will try to emit `[` `]` `.` for blank pixels and usually fail.

### 5.2 What Kraken *does* offer (and what it doesn't)

| Knob | Effect | Helps fragments? |
|---|---|---|
| `ketos segtrain --suppress-regions` | Train baseline-only, ignore region classes. | Yes — lets you fine-tune baselines on Sogdian without a region policy. |
| `--valid-baselines` / `--valid-regions` | Restrict training to specific line/region types. | Yes — excludes `Heading`/`Interlinear` noise from `Default`-line training. |
| `--bounding-regions` | Treat certain region types as hard line boundaries. | Yes — prevents polygons from straying across column edges. |
| `--augment` | albumentations-style line augmentation. | Neutral — no built-in lacuna/hole simulator. |
| `ketos pretrain` (Lacuna Reconstruction) | Self-supervised contrastive pretraining that masks line patches. | Helps *recognizer* robustness; does nothing for seg. Based on Vogler et al. NAACL Findings 2022 ([aclanthology.org/2022.findings-naacl.15](https://aclanthology.org/2022.findings-naacl.15/)). |
| `segment(mask=…)` (inference) | Page-level binary mask; 0-valued regions ignored. | Inference-only; suppresses seg in known-damaged bands. |
| **per-pixel "ignore" for training** | — | **Does not exist.** The seg dataset builder renders baselines as buffered polylines into the target tensor with no ignore channel. |

Refs:
[segtrain docs](https://kraken.re/6.0.0/training/segtrain.html),
[recognition dataset source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/recognition.py),
[segmentation dataset source](https://github.com/mittagessen/kraken/blob/9a218ce8/kraken/lib/dataset/segmentation.py).

### 5.3 Practical recipe for our repo

1. **Seg:** fine-tune BLLA on 100–300 Sogdian pages with baselines drawn per
   §4.1 and `DamageZone`s covering lacunae. Use `--suppress-regions` first
   (baseline-only), then add regions once baselines are stable.
2. **Rec:** fine-tune from the bundled Avestan model
   (`models/kraken/avestan_ms0040.mlmodel`) — out-of-domain but script-adjacent
   and cuts data needs per [`kraken-training-data-research.md`](./kraken-training-data-research.md).
   Drop lines whose polygon is >50% blank *before* training (a prep step in
   our `data/manifest.py` loader, not a Kraken feature).
3. **Inference:** ship a `mask.png` per folio for known damage, and pass
   `-m` to `kraken segment`.
4. **Experiment channel (off the main path):** try a dedicated lacuna token
   (`□` U+25A1) in recognition GT for lines with *small* internal gaps, and
   measure CER against the split-baseline baseline before adopting.

---

## 6. Open issues & unresolved debates

- **CurT / orli** ([github.com/mittagessen/orli](https://github.com/mittagessen/orli)):
  Bézier-curve line predictor that may replace BLLA. May handle fragmentary
  baselines differently. No release date. Re-evaluate when it lands.
- **No standard lacuna token for Kraken.** Leiden `[...]` is philologically
  standard but CTC-unfriendly. Community has not converged.
- **RTL × fragmentary interaction is unstudied** in Kraken literature. Watch
  for ordering bugs when split fragments of one logical row are placed in
  `ReadingOrder`.
- **Diplomatic vs. normalized GT.** Kraken docs say "diplomatic only"; the
  *Research Fragments* blog
  ([researchfragments.blogspot.com](http://researchfragments.blogspot.com/2023/08/escriptorium-lets-try-breaking-some.html))
  got good results training on normalized text. For Sogdian heterograms this
  is a real decision — pick one, document it, apply it uniformly.
- **No published Sogdian HTR annotation guidelines.** This repo is among the
  first; our convention in §4 should be treated as provisional and revised
  once we have measured CER on a held-out fragmentary folio set.

---

## 7. Sources

**Primary (Kraken / eScriptorium)**
1. Kiessling, B. (2020). "A Modular Region and Text Line Layout Analysis System." ICFHR 2020. [hal-04442992](https://hal.science/hal-04442992/document)
2. Kiessling, B. "Kraken — an Universal Text Recognizer for the Humanities." DH abstract. [dh-abstracts.library.virginia.edu/works/9912](https://dh-abstracts.library.virginia.edu/works/9912)
3. Kiessling, B. CurT presentation. [hal-05438436v1](https://hal.science/hal-05438436v1/file/kraken_ben.pdf)
4. Kraken docs: [kraken.re](https://kraken.re/) — segmentation, training, ATR (v5.1–7.0)
5. Kraken source: [github.com/mittagessen/kraken](https://github.com/mittagessen/kraken) — issues #745, #677, #773
6. BLLA base model card: [zenodo.org/records/14602569](https://zenodo.org/records/14602569)
7. eScriptorium docs: [escriptorium.readthedocs.io](https://escriptorium.readthedocs.io/en/latest/)
8. eScriptorium tutorial (OpenITI): [openiti.org/…/eScriptorium-Tutorial.pdf](https://openiti.org/assets/documents/eScriptorium-Tutorial.pdf)

**Secondary (adjacent research)**
9. Vogler et al. (2022). "Lacuna Reconstruction: Self-Supervised Pre-Training…" NAACL Findings. [aclanthology.org/2022.findings-naacl.15](https://aclanthology.org/2022.findings-naacl.15/) · arXiv [2112.08692](https://arxiv.org/abs/2112.08692)
10. "Mind the Gap: Analyzing Lacunae with Transformer-Based Transcription" (2024). arXiv [2407.00250](https://arxiv.org/html/2407.00250)
11. CATMuS guidelines: [catmus-guidelines.github.io](https://catmus-guidelines.github.io/) · HAL [hal-04346939](https://hal.science/hal-04346939)
12. Leiden Conventions: [en.wikipedia.org/wiki/Leiden_Conventions](https://en.wikipedia.org/wiki/Leiden_Conventions)
13. TEI P3 (primary sources): [tei-c.org/Vault/GL/P3/PH.htm](https://www.tei-c.org/Vault/GL/P3/PH.htm)
14. EpiDoc (gap/lacuna): [epidoc.stoa.org/gl/dev/trans-lostcharapprox.html](https://epidoc.stoa.org/gl/dev/trans-lostcharapprox.html)
15. Segmenting Dead Sea Scroll fragments (2024): arXiv [2406.15692](https://arxiv.org/html/2406.15692)
16. Research Fragments blog (normalized transcription experiment): [researchfragments.blogspot.com](http://researchfragments.blogspot.com/2023/08/escriptorium-lets-try-breaking-some.html)