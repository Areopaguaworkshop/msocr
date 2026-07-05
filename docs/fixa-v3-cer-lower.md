# Fix A v3: Pushing CER below 77.75%

Status: builds on `fix-a-v2-syriac-script-finetune.md` (SHIPPED, 77.75%
holdout CER). This is a set of further levers to try, ordered by
effort-to-impact ratio. None of these require reopening the "unfreeze the
backbone" question — that's a confirmed dead end at this data volume.

## Before anything else: two free checks

### 0.1 Verify the CER number itself isn't inflated by eval artifacts
Before spending compute chasing model improvements, confirm the 77.75%
isn't partly a measurement problem:
- Check the holdout GT and model output are in the same Unicode
  normalization form (NFC vs NFD) at eval time — a mismatch here silently
  counts correct characters as substitutions.
- Pull the per-character confusion matrix from the `ketos test` report
  (Correct-Generated table, same format kraken already prints — see the
  "Errors Correct-Generated" section in a standard `ketos test` run) and
  check whether errors cluster on the 5 new Sogdian classes (expected,
  fixable with more signal) or are spread across the 20 shared Syriac
  classes too (would suggest the frozen backbone itself is a weaker match
  for this manuscript's actual letterforms than assumed, which changes
  priorities — see 3.3).
- This costs zero GPU time and tells you where the 77.75% is actually
  coming from before you guess.

### 0.2 Confirm whether validation had actually plateaued
Val accuracy climbed 0.298 → 0.319 across 60 epochs and hit `max_epochs`,
not early stopping. That's a sign of **underfitting**, not convergence —
the run may have been stopped before it was done, not because it stopped
improving. This is the cheapest possible lever: rerun the identical 3.2
recipe with `--epochs 200 --lag 25` (no other changes) and see if it keeps
climbing. If it does, you get free CER improvement with zero new ideas.

## 1. Cheap architecture/hyperparameter levers (no new data needed)

### 1.1 Give the frozen backbone more capacity to work with — use `--append`
Kraken has a purpose-built mechanism for exactly this situation that wasn't
used in v1/v2: `--append <idx> -s '<vgsl spec>'` splits the loaded model at
a layer and appends new layers, auto-initializing them and auto-adding a
new output/CTC layer sized to the training alphabet. This is structurally
different from `--resize union --freeze-backbone`:

- `union` + frozen backbone = a single 42-way **linear** classifier reading
  directly off frozen Sophro features. The 22 shared-class rows keep their
  trained weights; the 5 new rows are a bare linear probe with no extra
  representational capacity of their own.
- `--append` lets you insert a **small new trainable recurrent layer**
  between the frozen backbone and the classifier — e.g. a compact
  bidirectional LSTM — that can learn to recombine the frozen Syriac
  features into whatever combination helps distinguish the 5 Sogdian
  classes, before a freshly-initialized classifier reads it. The backbone
  stays structurally frozen (not just LR-gated), so there's no reintroduction
  of the catastrophic-forgetting risk from 3.3 — the only thing training is
  new capacity, not old weights.

Concrete command to try (adjust the split index to the layer just before
the existing classifier/CTC layer — inspect the model summary printed at
`ketos train` startup to find it):

```
ketos train \
  -i c2av_finetune_union_frozen.safetensors \
  --append <idx> -s '[Lbx64 Do0.1]' \
  --resize union \
  --freeze-backbone 999999 \
  --warmup 200 --lr 1e-4 --augment \
  --min-epochs 15 --lag 15 --epochs 100
```

Run this as an ablation against 3.2 with identical data/epochs/LR budget.
If it beats 77.75%, the bottleneck was representational capacity of a bare
linear head, not just training signal — worth iterating on layer width
(32/64/128) from there. If it doesn't, the bottleneck is genuinely data
volume (see Section 2), and this avenue can be dropped.

### 1.2 Oversample the rare classes (zero new annotation, one file change)
qushshaya (192) and khaph (57) have reasonable counts; rukkakha (11), zhain
(10), and fe (10) do not — with CTC's per-line loss averaging, lines
containing rare classes contribute the same weight as lines that don't,
so the 5-10 lines containing a rukkakha/zhain/fe are a tiny fraction of
gradient signal across 179 lines. Duplicate those specific lines 3-5x in
the training file list (not the validation/holdout list) to boost their
effective sampling frequency. This is a training-list change only, no code
or architecture change, and directly targets the classes most likely still
undertrained (rukkakha/zhain/fe each ~5-10x rarer than the median class).

### 1.3 LR/epoch sweep specific to the frozen-backbone regime
Because the backbone is frozen, this regime is safe to push harder than
runs #3/#4 (those failed because of the unfrozen backbone, not the LR
itself). Try 3e-4 and 5e-4 in addition to the existing 1e-4, each for the
full 100+ epochs from 0.1, since a bare/small trainable head can typically
tolerate higher LR than a full network fine-tune. Keep whichever combination
of (1.1 layer width) x (this LR) x (0.2 epoch budget) gets the best val CER,
then confirm on holdout only once, to avoid overfitting hyperparameter
choices to the 19-line holdout itself.

## 2. Data volume levers (higher effort, likely the real ceiling-raiser)

### 2.1 Synthetic data via the SogdianOCR CycleGAN ink-aging pipeline
The CycleGAN ink-aging synthesis and Pango-based line rendering already
built for SogdianOCR v0.7 solves exactly the data-scarcity problem here.
Render synthetic Sogdian-script line images from any available digitized
Sogdian text (dictionary entries, other Turfan fragment transcriptions,
even generated pseudo-words using the known phoneme inventory) targeting
words/sequences that contain rukkakha, zhain, and fe specifically, then
age them with the CycleGAN bridge to match the c2av plates' degradation
profile. Mix these into training only (never into val/holdout, to keep
evaluation honest) alongside the 179 real lines. This is the most direct
fix for "148-179 real lines is at the bottom of the viable range" — it
doesn't need new manuscript photography or transcription, just reuses
tooling that already exists for the sibling project.

### 2.2 Prioritized real annotation
If more Berlin Turfan Christian Sogdian plates are available, prioritize
annotating ones likely to contain more rukkakha/zhain/fe occurrences over
ones that mostly repeat already-common letters — check catalog descriptions
or do a quick visual scan for diacritic density before committing transcription
time.

## 3. Inference-time levers (no retraining, apply to the existing 77.75% model)

### 3.1 Lexicon-based post-correction
Christian Sogdian Turfan texts are translations of known Syriac/Greek
liturgical and biblical originals — meaning the underlying vocabulary is
closed and largely attested in published Sogdian dictionaries and text
editions (Gharib, Sims-Williams editions, etc.). After the existing
`syriac_to_latin.py` transliteration step, add an edit-distance-based
lexicon correction pass: for each output word, if it's within 1-2 edits of
a known Sogdian word and the model's word isn't itself attested, substitute
the dictionary match. This is a classic, low-risk OCR post-processing
technique and can meaningfully lower word-level CER without touching the
model at all. Build the lexicon from whatever digitized Sogdian
dictionary/corpus text is accessible.

### 3.2 Beam search instead of greedy decoding
Kraken exposes `kraken.lib.ctc_decoder.beam_decoder` as an alternative to
the default greedy decoder. Beam search with a small beam (3-5) sometimes
recovers correct sequences that greedy decoding misses at ambiguous
timesteps, especially for the still-underfit rare classes. Essentially
free to try against the existing model — no retraining, just swap the
decoder at inference time and re-run the holdout eval to see if it moves
the number at all.

### 3.3 Only if 0.1 shows errors spread across shared Syriac classes too
If the confusion-matrix check in 0.1 shows the model is also
misrecognizing the 20 *shared* Syriac letters (not just the 5 new ones),
that's a different problem than anything above: it would mean the frozen
Sophro backbone's features, trained on whatever corpus/hand it originally
saw, don't transfer as cleanly to this specific manuscript's handwriting as
assumed. That reopens the question of whether frozen-backbone-forever is
really optimal, but the fix there is different from unfreezing blindly —
it would mean the base model itself may need a light full fine-tune on
Syriac-only material *before* the Sogdian delta is added, so the backbone
adapts to this hand while still learning on an alphabet where you have
lots of comparison data. Don't pursue this unless 0.1 shows it's needed.

## Suggested order of execution

1. 0.1 and 0.2 (free, do first, may already resolve part of the gap)
2. 1.2 (oversampling — cheap, one file edit)
3. 1.1 (`--append` ablation — the most promising untried lever)
4. 1.3 (LR/epoch sweep, combined with whatever 1.1 lands on)
5. 3.1 and 3.2 (inference-time, apply regardless of what 1-2 find, stack
   with whichever training improvements land)
6. 2.1 (synthetic data) if 1's ceiling is reached and more CER reduction
   is still needed — this is the biggest effort but also the biggest
   likely ceiling-raiser given 179 lines is inherently thin for 5 new
   classes

## What to keep unchanged from v2

- `--resize union` (preserving shared-class weights is confirmed useful)
- Frozen backbone (unfreezing is a confirmed dead end at this data volume)
- The Latin/Syriac mapping decisions (t/θ, g/γ, x→khaph, etc.) — don't
  relitigate the alphabet while isolating these variables, or results
  across experiments won't be comparable
