# Fix A v6: Script-Tradition-Aware Plan

Status: builds on `fix-a-v5-plan.md`. This update doesn't replace v5's core
levers (LOOCV, augmentation, targeted annotation) — those are still valid
and unaffected by anything below. It replaces v5's specific recommendation
of Vienna GT as the intermediate-adaptation dataset, adds a free
segmentation sanity check, and adds an empirical pre-check step before any
more training investment.

## What changed since v5

1. **Vienna GT (ÖNB Cod. Syr. 1) is Serto script.** Christian Sogdian
   Turfan manuscripts descend from the Church of the East / Eastern Syriac
   (Estrangela-derived) tradition, not Serto. Serto and Eastern-Syriac
   letterforms genuinely differ in shape for several shared letters (kaph,
   lamadh, meem/simkath documented as differing). Pretraining on Vienna GT
   risks reinforcing the wrong stroke shapes for your 20+ shared classes,
   not just failing to help. Downgraded from "worth trying" to "lower
   priority than the alternative below."
2. **A better-matched public dataset exists**: HTR Winter School 2025 — MS
   Jerusalem, Saint Mark's Monastery 36 (Zenodo 10.5281/zenodo.18157525),
   described as primarily Estrangelo with Eastern features, 12th-14th
   century, 133 bifolios. Confirmed downloadable directly (plain Zenodo
   zip files — `alto.zip` 2.6MB, `page.zip` 2.5MB, `images.zip` 970MB —
   no Git LFS gotcha, CC-BY-4.0). Right script family, at least. Same
   caveat as before applies twice over though: it's a formal codex hand,
   not the cursive documentary hand your Turfan fragments show, and its
   transcription convention **excludes qushoyo and rukokho** — same as
   Vienna GT — so it only helps with 3 of your 5 new classes
   (zhain/khaph/fe), not qushshaya/rukkakha.
3. **Sophro Mhiro's actual training data is documented** (Zenodo
   10.5281/zenodo.17406773): trained on a deliberate blend of Serto,
   Estrangela, and Eastern Syriac — 38 manuscripts, 6th-20th century,
   ~91,000 characters / 3,466 lines, mostly biblical text. So it's not
   purely wrong-tradition, but it's a generalist blend, not anything
   specialized for Eastern-Syriac-descended documentary hands. Its ground
   truth also explicitly **excluded "extra letters for Garshuni or
   Syro-Persian"** — i.e., any letters added to write a non-Syriac
   language in Syriac script, which is structurally the same category your
   5 Sogdian-specific letters fall into. Confirms (with a source now) that
   those 5 classes are genuinely starting from zero exposure in Sophro,
   regardless of which further base you pick.
4. **A separate, unrelated finding worth checking for free**: a March 2026
   real-world evaluation of Sophro Mhiro on an out-of-distribution 8th
   century Estrangela manuscript found CER dropped from 28% to ~6% purely
   by fixing line segmentation — same recognition model, same everything
   else. Segmentation quality, not recognition capacity, was the dominant
   error source in that case.

## Step 0: Two free checks before any more training

### 0.1 Segmentation pipeline audit
Confirm every CER number reported in v2-v5 came from `ketos test`/`ketos
train` running against your manually-annotated PAGE-XML line polygons
throughout — not from any step that re-derives line boxes via automatic
segmentation. This should already be true (GT-based fine-tuning normally
isolates recognition from segmentation by construction), but it's worth an
explicit one-line check given how irregular and torn the c2av fragments
are — if segmentation ever entered the eval chain anywhere, that's a
completely separate CER contributor from anything discussed in v1-v5, and
given the referenced case saw a 22-point swing from segmentation alone on
a comparatively well-preserved codex, an irregular Turfan fragment could
show an even larger effect.

### 0.2 Confirm the diacritic gap can't be closed with existing public data
Before building anything on Jerusalem MS 36, it's worth a quick check
(one search, not a project) for whether any published Syriac GT dataset
transcribes qushoyo/rukokho at all — both public options found so far
exclude them by convention, which may just be the standard convention for
this kind of transcription project rather than a coincidence. If no such
dataset exists, that's useful to know now: it means qushshaya/rukkakha
will always be a 148-lines-only problem, and effort is better spent
getting more real c2av annotation for exactly those two marks (they're
already your rarest classes at 11 and a small handful of occurrences)
rather than searching for external GT that isn't going to appear.

## Step 1: LOOCV (carried from v5, unchanged priority)

Still first for any training-affecting change: run leave-one-plate-out CV
across all 12 c2av plates on the existing v3 0.2 recipe before trusting
any further deltas, including whatever comes out of Step 2 below. Nothing
in this update changes why that's necessary.

## Step 2: Intermediate adaptation — revised recommendation

If 0.1/0.2 don't reveal a cheaper fix, and only after Step 1 gives a
trustworthy baseline distribution to compare against:

1. Fine-tune Sophro Mhiro's backbone on Jerusalem MS 36 instead of Vienna
   GT — same mechanism as v5's proposal, better-matched script family.
2. **Empirically verify before committing further training time**: crop a
   few plain-Syriac-only lines from c2av images (letters shared across all
   traditions, no Sogdian-specific ones), and run zero-shot inference with
   (a) base Sophro Mhiro and (b) the Jerusalem-MS36-adapted model. Compare
   CER on just those crops. Only proceed to fine-tune the Sogdian delta on
   top of whichever wins — don't assume the theoretical script-family
   argument above will actually pay off without checking.
3. Temper expectations going in: even in the best case this addresses
   script-family mismatch and 3 of 5 new classes. It does not address the
   qushshaya/rukkakha diacritic gap (per 0.2) or the formal-book-hand vs.
   cursive-documentary-hand gap raised earlier in this conversation. Real
   annotation targeting those two remains the only lever with no
   substitute.

## Step 3: Augmentation + synthetic data (carried from v4/v5, unchanged)

Heavier `ocropy_degrade`/`distort_line` augmentation of the real 148 lines,
and `ketos linegen` synthetic generation once font glyph coverage for
zhain/khaph/fe is confirmed. Still valid, still secondary to Step 2 for the
same reason as before — these expand visual diversity around a feature
space, Step 2 (if it works) changes what the feature space represents.

## Step 4: Targeted real annotation (carried from v5, unchanged, now more clearly the one lever nothing else substitutes for)

Given 0.2 and Step 2's tempered expectations both point at the same gap —
qushshaya/rukkakha have no public data source and Sophro never saw
anything like them — this is the most clearly irreplaceable lever in the
whole plan. Prioritize any further annotation specifically toward lines
containing these two marks over general volume.

## Suggested order

1. §0.1 and §0.2 — free, five minutes, do first
2. §1 LOOCV — cheap, decisive, needed to judge everything after it
3. §2 intermediate adaptation on Jerusalem MS 36, with the empirical
   zero-shot check before committing to a full fine-tune
4. §3 augmentation/synthetic, layered on whichever backbone wins §2
5. §4 targeted annotation — ongoing in parallel with all of the above,
   highest ceiling long-term, no technical substitute

## What this rules out / de-prioritizes

- Vienna GT (ÖNB Cod. Syr. 1) as the intermediate-adaptation dataset —
  wrong script tradition, kept as a fallback only if Jerusalem MS 36 turns
  out inaccessible or the empirical check in §2.2 favors it unexpectedly.
- Assuming any intermediate-adaptation dataset solves the full 5-class
  problem — none of the public options transcribe qushshaya/rukkakha, so
  at best this is a partial fix (3/5 classes plus general feature
  adaptation), not a complete one.
