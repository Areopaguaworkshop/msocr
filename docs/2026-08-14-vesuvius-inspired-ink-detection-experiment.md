# Vesuvius-inspired ink detection experiment for fragmented manuscripts

**Date:** 2026-08-14  
**Status:** proposed development experiment; not part of the production HTR path  
**Decision:** run a small 2D ink/damage segmentation experiment only after a held-out evaluation set is prepared. Do not add it to the default `preprocess`, annotation, or HTR commands unless it improves the agreed measures.

## Executive summary

The Vesuvius Challenge ink-detection loop is relevant to MSOCR because both
projects face scarce labels, damaged material, and a human-in-the-loop path to
better training data. Its useful idea is not its CT model architecture: train a
pixel classifier, review its strongest candidate regions, turn corrections into
new labels, and repeat.

MSOCR should adapt that idea to ordinary 2D manuscript images as an optional
**ink / substrate / damage** mask. The mask may improve fragment isolation,
binarisation, and annotation visibility. It must not be treated as text
recognition, a replacement for Kraken, or evidence for a character unless a
human palaeographer confirms it.

The current production path remains:

```text
publication plate -> isolate -> binarise -> deskew -> line geometry -> Kraken HTR
```

The experiment is an upstream sidecar:

```text
processed page -> experimental mask -> human review -> approved mask/labels
                         |                                  |
                         +-> optional display/crop aid       +-> next model round
```

This deliberately does not import Vesuvius's 3D CT/TimeSformer stack. MSOCR's
input is already a visible 2D page image, so a small 2D segmentation baseline
is the correct first test.

## What has been verified

The Vesuvius Challenge currently identifies ink detection as an open machine
learning and domain-generalisation problem. Its public description says models
are trained on visible-ink fragments, inferred on other material, and improved
through iterative pseudo-labelling. It also lists the `First Letters` prize
track and a 25 June 2027 deadline. [Official Vesuvius Challenge site](https://scrollprize.org/)

Their winning approach used repeated label expansion and cleaning (reported as
roughly 15 rounds), but it was built for volumetric X-ray CT using models such
as TimeSformer and 3D ResNets. [Winner's technical overview](https://github.com/younader/Vesuvius-Grandprize-Winner)

Paul Henderson is publicly listed on the Vesuvius technical team. The public
site supports monitoring its news and prize pages; this report does not rely
on the unverified claim that there are no roles available, nor on an
unpublished blog post. [Vesuvius team and news](https://scrollprize.org/)

## Fit to MSOCR

| Vesuvius problem | MSOCR analogue | Transferability |
|---|---|---|
| Carbon ink versus papyrus in 3D CT | Faint ink, stains, cracks, bleed-through, substrate, and margins in 2D images | Method transfers; data/modality do not |
| Surface-segment inference | Isolated manuscript-fragment image | Partial |
| Iterative pseudo-label review | Annotation review and active selection of useful page regions | Strong |
| Readability/letter discovery | Better masks, line crops, and ground truth for Kraken | Strong, with human validation |
| Virtual unwrapping | No direct MSOCR equivalent | None; out of scope |

The proposed model answers only: **which pixels are likely usable ink, blank
substrate, or damage/noise?** It does not identify Syriac characters,
determine RTL order, reconstruct missing text, or resolve a philological
reading.

## Existing integration points

The experiment can use existing MSOCR boundaries without changing the runtime
contract:

- [`ManuscriptPreprocessor.preprocess_single`](../msocr/preprocessing/preprocessor.py)
  already produces enhanced and binary images after normalisation, denoising,
  contrast enhancement, and deskewing.
- [`SessionManager.create_session`](../msocr/data/session_manager.py) and the
  annotation API already provide a session-local annotation workflow; masks
  should first be stored beside a session as optional artifacts.
- [`_resolve_image_for_xml`](../msocr/training/orchestrator.py) demonstrates
  that PAGE XML remains the project’s training hand-off. Existing line
  transcripts must remain untouched.
- [`predict`](../msocr/models/inference.py) is Kraken line recognition.
  Experimental masks may help generate/display a crop, but must not silently
  change a production recognition request.

This aligns with the existing fragment-segmentation plan: learned assistance
may improve geometry and annotation speed, while human-confirmed reading order
and gap metadata remain outside any image model.

## Minimal experiment design

### Labels

Annotate a modest, diverse sample of already-isolated fragments at pixel or
coarse-polygon level. Use three mutually exclusive labels:

1. `ink` — visible manuscript strokes worth retaining.
2. `substrate` — parchment/paper/papyrus background, including benign texture.
3. `damage` — holes, tears, rulers, plate labels, severe stains, bleed-through,
   or regions where the image cannot support a reliable ink decision.

Keep `uncertain` as an annotation-review state rather than a training class;
it is excluded from loss and from automatic use. This prevents a model from
learning the annotator’s uncertainty as either ink or background.

Use a manuscript- or plate-level split, never random crops from the same plate
in both training and holdout. This is necessary because texture, conservation
damage, and photography conditions otherwise leak into the evaluation.

### Baseline model

Start with one lightweight 2D semantic segmentation model (for example, a
small U-Net using already-installed PyTorch/OpenCV dependencies if available).
No new dependency and no architecture comparison in round one. Train on image
tiles from the **enhanced but not hard-binarised** image so the model sees
faint-stroke intensity information.

Output a probability map, not a forced binary image. The user-facing dev
option can render a thresholded overlay, but threshold selection belongs to
evaluation and must be recorded with the model.

### Iterative review loop

```text
Round 0: hand-label diverse plates -> train baseline -> evaluate holdout
Round 1: infer only on new, unlabeled plates -> rank uncertain/high-value tiles
Round 2: palaeographer corrects selected tiles -> add approved labels -> retrain
Stop: no material held-out gain or annotation time is better spent on transcripts
```

Select review tiles from two buckets:

- high ink probability in regions the classical preprocessing discarded, to
  recover possible faint writing;
- boundary/uncertain probability regions, to correct false positives from
  stains, tears, and bleed-through.

Do **not** train directly on model predictions. Pseudo-labels only become
training labels after human approval. In a small, specialised corpus this is a
safer adaptation than self-training on unreviewed output.

## Development-only interface

After the baseline clears its first gate, expose one explicit option, for
example:

```bash
uv run msocr preprocess --ink-mask experimental /path/to/images
```

It writes sidecar artifacts only, such as:

```text
processed/page_001_enhanced.png       # existing artifact
processed/page_001_ink-probability.png
processed/page_001_ink-overlay.png
processed/page_001_ink-mask.json      # model ID, threshold, source checksum
```

The default command stays unchanged. The annotation UI may later offer a
non-destructive overlay toggle; it must clearly label the layer as
`experimental prediction` and preserve the original image as the source of
truth.

No runtime/API mask field is proposed in the first implementation. A mask that
cannot demonstrate value in annotation or held-out recognition does not need a
production interface.

## Evaluation gates

Pixel scores alone are insufficient: the point is better HTR data and less
human effort. Freeze a small plate-level holdout before the first training run
and report all of the following by manuscript/source group:

| Gate | Measure | Minimum to proceed |
|---|---|---|
| Mask validity | ink-class precision and recall on human labels | beats Sauvola/current binary baseline on both, with no major damage false positives |
| Annotation utility | median minutes to prepare a page or accept/correct a line | at least 20% reduction without lower review quality |
| Crop utility | usable line-crop rate after segmentation | improves over current preprocess-only output |
| Recognition utility | frozen Kraken holdout CER, same split/model training budget | improves, or is statistically indistinguishable while saving meaningful annotation time |
| Safety | false positive ink in damage/labels/rulers | manually reviewed and below a pre-agreed tolerance |

CER is the decisive downstream metric, using the existing fixed holdout and
confusion-matrix workflow. The previous reduction from 82% to 71% is useful
context, but it is **not** a target that a segmentation model may claim without
an independently frozen comparison.

## Implementation plan

### Phase 0 — dataset and baseline (one small milestone)

1. Choose 20–30 representative isolated plates/fragments across source,
   damage level, ink density, and image quality.
2. Create a compact review/export format for the three labels. Reuse the
   annotation session’s page coordinates; do not create a second annotation
   system.
3. Freeze plate-level train/validation/holdout manifests and save source-image
   checksums.
4. Measure current preprocessing: binary-mask precision/recall on holdout,
   usable crop rate, annotation time, and CER baseline.

**Stop condition:** insufficient label agreement or a holdout too small to
distinguish plate-specific texture from generalisation. Improve the label guide
instead of building a model.

### Phase 1 — offline segmentation prototype

1. Add an offline training/inference script under an experimental namespace;
   it reads frozen labels and writes a probability map.
2. Add exactly one small test covering artifact shape/metadata and reject a
   source checksum mismatch.
3. Produce overlays for the holdout; conduct blind palaeographic review.
4. Compare against the current enhanced/binary preprocessing baseline.

**Stop condition:** mask quality does not clear the pixel and damage-safety
gates. Keep the labels and stop; do not wire the model into the CLI.

### Phase 2 — development-only preprocessing option

1. Add the explicit `--ink-mask experimental` mode only if Phase 1 passes.
2. Save model/threshold/checksum metadata with sidecar artifacts.
3. Add an annotation overlay toggle, disabled by default, if reviewers find
   the overlay useful.
4. Run timed annotation and crop-quality comparison on new plates.

**Stop condition:** no 20% annotation-time improvement and no crop/recognition
gain. Retain the offline tool for research; do not make it a supported path.

### Phase 3 — decision on supported use

Promote only the narrow, proven use:

- `annotation aid` if it speeds accurate human ground-truth creation;
- `preprocessing candidate` if it improves held-out CER/crops;
- neither if it only produces visually convincing overlays.

Production promotion requires explicit user choice or a documented decision,
model versioning, a failure fallback to the current classical pipeline, and a
repeatable holdout report.

## Risks and controls

| Risk | Control |
|---|---|
| Treating a stain or bleed-through as ink | explicit damage class; human approval before labels/transcripts are changed |
| Data leakage from crops of one plate | group split by plate/manuscript before tiling |
| Model makes annotations look authoritative | UI label and opacity overlay; original image always remains visible |
| HTR CER worsens after mask use | retain enhanced original and compare on the frozen split; never overwrite input |
| Scope expands into a second OCR system | no character classifier, language model, unwrapping, or architecture bake-off |
| GPU cost exceeds evidence | prototype locally or in one bounded remote run; scale only after Phase 1 gate |

## Recommendation

Implement Phase 0 first. It is a low-risk way to test the Vesuvius insight
against MSOCR's actual bottleneck: trustworthy annotated ground truth from
fragmented pages. The experiment is worthwhile if it makes human annotation
or downstream CER measurably better; otherwise, more direct line/transcript
annotation remains the better investment.
