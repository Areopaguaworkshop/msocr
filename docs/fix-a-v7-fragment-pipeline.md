# Fix A v7: Full Fragment-Manuscript OCR Pipeline (Target: CER < 20%)

Status: this is a full-pipeline spec, not a single lever like v1-v6. It
covers everything from raw fragment photograph to final corrected
transcription, with concrete kraken/eScriptorium tools and options at each
stage. It absorbs everything validated in v1-v6 (union+frozen recognition
fine-tune, LOOCV, script-family-aware intermediate adaptation, targeted
annotation) and adds the two pieces that were never addressed:
**segmentation** and **inference-time preprocessing**, both of which
directly answer this request.

## Stage 0: Realistic goal-setting, stated plainly

Before the pipeline — what does the literature say sub-20% CER actually
requires, so the plan can be judged against real numbers rather than hope:

| Material | Typical CER (published/vendor figures) |
|---|---|
| Well-preserved script, matched model, large training set (Gothic textura, Latin liturgical hands) | 3-5% |
| Damaged parchment / unusual hands, public model | 10-15%, "improves substantially with custom training" |
| Damaged parchment / unusual hands, custom-trained model (thousands of lines available) | can approach the 3-5% range |
| Non-Latin, degraded, severely underrepresented scripts (the category your material falls in) | published surveys note performance "drops sharply" outside clean-Latin-script conditions, with few benchmarks even attempted |

Every one of the low-CER numbers above assumes a well-resourced target:
thousands of training lines, an established script the base model was
built for, and — critically — layout that isn't physically fragmentary.
Your situation stacks four separate hard problems that none of those
benchmarks combine: a cursive documentary (not book) hand, a script
tradition your base model wasn't specialized for, a torn/irregular
physical fragment, and 5 characters with ~150 lines of data total. Sub-20%
CER is a reasonable **staged target**, not a guarantee from pipeline
engineering alone — the biggest lever remains data volume (v5/v6 §4), and
everything below is about not leaving CER on the table elsewhere while
that data grows.

## Stage 1: Image preprocessing (before segmentation)

### Tools and options

| Step | Tool options | Notes |
|---|---|---|
| Binarization | `kraken.binarization.nlbin` (ocropus-derived, kraken's built-in default); scikit-image `threshold_sauvola`; Tensmeyer & Martinez FCN-based binarization (DIBCO-competition-winning neural method, validated on degraded material including palm-leaf manuscripts) | Modern kraken's neural segmenter (blla) can run on grayscale directly — binarization is now optional for segmentation, not mandatory. A controlled study on degraded historical prints found binarization alone outperformed deblurring alone, and stacking multiple enhancement methods added little beyond one good binarization choice — don't over-engineer this stage. |
| Contrast enhancement | OpenCV CLAHE (`cv2.createCLAHE`) | Useful for low-contrast/faded ink specifically, cheap to test. |
| Bleed-through / staining removal | Dedicated preprocessing stage recommended in the modular-pipeline literature before OCR/post-correction, rather than folding it into binarization | Given the mottling/staining visible in the c2av plate you shared, worth testing as a discrete step — even a simple background-flattening pass (subtract a heavily-blurred copy of the image from itself) can help before binarization. |
| Deskew | Standard projection-profile or Hough-transform deskew (OpenCV/ImageMagick `-deskew`) | Matters more for baseline detection accuracy than for recognition itself — a skewed fragment can throw off baseline fitting even when the letters themselves are legible. |

### Recommendation for your specific case
Test exactly two binarization options empirically (nlbin vs. Sauvola)
against a handful of representative c2av crops, pick whichever preserves
diacritic dots and thin strokes better (qushshaya/rukkakha and the
Sogdian-specific letters are diacritically dense — coarse thresholding can
merge or erase small marks), and stop there. Per the literature finding
above, a longer preprocessing chain is unlikely to buy much once one good
binarization choice is made. Given the print-reproduction origin of the
plate you shared (possible ink retouching noted earlier), also test
leaving the image **unbinarized** and feeding grayscale directly into
segmentation — worth a direct comparison since kraken's neural pipeline
supports it and retouched-ink material may binarize unpredictably.

## Stage 2: Annotation policy for fragments specifically

This is the piece most fragment-OCR guides skip, and it's the direct
prerequisite for Stage 3.

- **Mark lacunae/damaged regions explicitly, don't just omit them.**
  eScriptorium's region/line typing supports marking areas as
  non-text or damaged; do this consistently for illegible or missing
  portions of each fragment rather than silently cropping around them —
  otherwise your segmentation model will be trained on inconsistent
  boundary conventions (sometimes a torn edge means "line ends here,"
  sometimes it means "line continues off-fragment," and the model can't
  tell those apart unless you annotate the distinction).
- **Decide a policy for physically disconnected fragment pieces on the
  same plate** (the c2av19-style images with a separate floating chunk of
  text detached from the main block) — treat each physically separate
  piece as its own region with its own reading-order position, annotated
  explicitly, rather than letting automatic reading-order inference guess
  at how disconnected pieces relate. This avoids the segmentation model
  learning to "bridge" gaps that shouldn't be bridged.
- **Baseline-only vs. baseline+region annotation**: given c2av's layout is
  a single column of running text (no columns, no marginalia, no complex
  regions), a baseline-only segmentation model is likely sufficient and,
  per kraken's own segmentation documentation, tends to be more robust and
  data-efficient than a full region+baseline model when training data is
  scarce — worth deliberately choosing the narrower target rather than
  training for layout complexity you don't have.

## Stage 3: Segmentation model fine-tuning (the piece not yet addressed in v1-v6)

### Why this matters specifically for your material
Kraken's default segmentation model (`blla.mlmodel`) is explicitly
documented by its own maintainers as designed for **non-fragmentary**
handwritten/printed pages, with known accuracy problems on **highly
diacritized scripts** and on **non-Latin baselines** generally — its
training corpus (cBAD) skews heavily toward Latin-script material. Your
material is fragmentary, highly diacritized (Sogdian needs qushshaya/
rukkakha marked precisely), and non-Latin. That's three separate
documented weak spots of the default model stacking on one input. This is
a very plausible independent contributor to your CER numbers, entirely
separate from anything the recognition-side experiments (v1-v6) could
have caught, since those all trained/evaluated against your own
hand-corrected line polygons — meaning **if your production pipeline
(as opposed to your ketos train/test evaluation) ever runs the default
blla segmenter on new, unannotated c2av-like fragments, that step could be
introducing errors your holdout numbers never saw.**

### Concrete fine-tuning approach
```
ketos segtrain -f page -N 100 -q early --min-epochs 50 \
  --suppress-regions \
  -i blla.mlmodel \
  -o c2av_segmenter \
  train_gt/*.xml
```
- `--suppress-regions` trains a baseline-only model per Stage 2's layout
  assessment — narrower target, more robust with your ~150-line scale of
  annotated material.
- Start from `blla.mlmodel` as the base (transfer learning, same logic as
  the recognition side) rather than training from scratch — with this
  little data, the same catastrophic-forgetting-vs-underfitting balance
  from the recognition experiments likely applies here too, so treat
  freeze-schedule tuning here as its own small ablation, not an
  afterthought.
- Verify actual flag support against your installed kraken 7.0.2 before
  running — `ketos segtrain --help` — given you've already found real
  discrepancies between kraken's documented and dispatched options on the
  recognition side (`--resize add/both` vs the working `union/new`), don't
  assume older tutorial flags transfer unchanged to your version.
- Evaluate with `ketos segtest` (kraken 7.0's segmentation-specific test
  command, which now reports baseline detection metrics directly rather
  than only pixel accuracy/IoU) against a held-out set of your own
  fragments, the same LOOCV-style discipline as v5/v6's recognition
  evaluation.

## Stage 4: Recognition model fine-tuning

Unchanged from v1-v6 — this doc doesn't relitigate that work. Carry
forward: `--resize union --freeze-backbone 999999`, LOOCV across all 12
plates (v5 §1), the script-family-aware intermediate adaptation candidate
from v6 (Jerusalem MS 36, empirically verified before committing), and
targeted annotation toward qushshaya/rukkakha specifically.

## Stage 5: Inference-time pipeline (the direct answer to "after fine-tuning, how do I clean images before OCR")

This is the ordered pipeline to run a **new, unannotated** fragment image
through end to end, and the one you should also use to generate the
crops that feed CER evaluation, so your evaluation numbers reflect the
same pipeline production will actually use:

```
raw fragment photo
  → Stage 1 preprocessing (binarization choice, CLAHE if faded, deskew)
  → Stage 3 fine-tuned segmenter (c2av_segmenter, baseline-only)
  → line polygon extraction + crop
  → Stage 4 fine-tuned recognizer (union+frozen model)
  → Stage 6 post-correction (below)
  → final transcription
```

Run this exact chain — not manual/GT-based line crops — for any CER
number you intend to report as "production performance," since manually
supplying correct line crops (as `ketos test` does against GT) measures
recognition alone and can look meaningfully better than what a real
end-to-end run on a new fragment produces if segmentation is the weaker
link (per Stage 3's reasoning, and per the March 2026 Sophro Mhiro
real-world evaluation that saw CER swing from 28% to 6% on segmentation
quality alone, same recognition model both times).

## Stage 6: Post-OCR correction

Two real options, with a note on why the one you already tried failed:

- **Lexicon/dictionary correction (tried in v3 §3.1, made CER worse)**:
  word-level edit-distance correction against a Sogdian vocabulary. This
  failed because your dominant error mode is deletions (CTC dropping
  characters, especially around the new diacritic classes), and a
  word-lookup correction step can't reconstruct characters that were never
  emitted in the first place — it can only choose among words built from
  what's already there. Not recommended to revisit as-is.
- **Learned sequence-to-sequence post-correction (ByT5 or similar
  byte-level transformer, not yet tried)**: fine-tune a small seq2seq
  model on (raw OCR output → corrected transcription) pairs from your own
  training lines. Unlike lexicon lookup, a seq2seq corrector operates over
  the full character sequence and can learn to *insert* characters the
  recognizer dropped, not just choose among existing ones — directly
  targeting your actual dominant error mode instead of a different one.
  This has published precedent specifically paired with kraken output in
  the early-printed-book HTR literature, for the same
  reason: catching systematic OCR error patterns after the fact. Feasible
  at your data scale since ByT5-small fine-tunes reasonably on a few
  hundred line pairs, though expect it to need augmented/duplicated pairs
  given how little data you have, similar to the recognition side.

## Stage 7: Full tool/option matrix (single reference)

| Stage | Tool | Key options |
|---|---|---|
| Binarize | `kraken.binarization.nlbin`, scikit-image Sauvola, Tensmeyer FCN binarizer | Pick one, verify diacritic-dot preservation, stop |
| Contrast | OpenCV CLAHE | Only if ink is faded, not blotchy |
| Bleed-through | Background-flatten (blur-subtract) as discrete pre-binarization step | Test given the mottling visible in your plates |
| Deskew | OpenCV/ImageMagick projection-profile deskew | Matters most for baseline fitting |
| Segment | `ketos segtrain` fine-tune of `blla.mlmodel`, `--suppress-regions` | Base-model transfer + freeze ablation, same discipline as recognition |
| Segment eval | `ketos segtest` | Baseline detection metrics, kraken 7.0 |
| Recognize | `ketos train --resize union --freeze-backbone 999999` (v3-v6 recipe) | Unchanged |
| Recognize eval | `ketos test`, LOOCV across all 12 plates | v5 §1 |
| Post-correct | ByT5 seq2seq fine-tune on (raw→corrected) line pairs | Targets deletions directly, unlike lexicon |

## Path to sub-20% CER: staged, not one-shot

1. **Near-term (pipeline fixes, no new data)**: Stage 1-3 segmentation
   work plus Stage 6 seq2seq post-correction, layered on the existing
   union+frozen recognizer. This won't get you to 20% alone, but it
   removes error sources (segmentation quality, uncorrectable deletions)
   that were never isolated from the recognition-model numbers in v1-v6 —
   expect this to meaningfully tighten the gap between your `ketos test`
   numbers and true end-to-end production numbers, and plausibly improve
   both.
2. **Medium-term (v6's data levers)**: script-family-matched intermediate
   adaptation (Jerusalem MS 36, empirically gated), heavier augmentation,
   more targeted annotation toward qushshaya/rukkakha specifically.
3. **Longer-term (the actual path to sub-20%)**: enough real annotated
   c2av-and-related-plate data that the comparison class shifts from
   "5-15 lines per rare class" to the hundreds-of-lines-per-class range
   that even the *hard* end of the published CER benchmarks above assumes.
   There's no pipeline engineering substitute for this — it's the same
   conclusion v1-v6 kept arriving at from different angles, now with a
   fuller picture of everywhere else CER could still be leaking from.
