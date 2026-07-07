# Fix A v5: Rethinking the Approach After §1's Falsification

Status: builds on `fixa-v4-result-2026-07-06.md` (§1 freeze-rows FAILED,
holdout regressed to 80.09%). This doc answers two direct questions first
(train/val/test split status, and how many lines is "enough"), then lays
out three-plus concrete levers for v5, reordered by what the evidence now
supports.

## Q1: Did you do a proper train/validation/test split?

Yes — this has been a real three-way split since v2, not a two-way
train/eval split: 10 c2av plates for training, 1 plate (c2av11) held out
as validation, 1 plate (c2av12) held out as test/holdout, never mixed.
That part of the methodology is sound.

The problem isn't the *existence* of the split, it's the **size and
granularity** of val/test: 5 validation lines and 19 holdout lines (each
a single plate) is thin enough that v4's own report flagged the
best-checkpoint selection as possibly noise-driven, and the −8.66-point
"failure" of §1 could partly be an artifact of which one plate happened to
be held out, not a real property of the freeze-rows idea.

## Q2: How many lines would get a better/more reliable result?

Two separate questions hide inside this — how much data to *train* on,
and how much to *hold out for evaluation* — and the literature on
low-resource HTR fine-tuning has fairly concrete numbers for both.

**Training volume:** A 2023 study on single-writer HTR fine-tuning found
that training a recognition model from scratch <cite index="31-1">did not converge when fewer than 230 lines were used for training</cite>,
which lines up with your own earlier feasibility research (800-2000+ for
from-scratch). But the same study found that *fine-tuning a well-chosen
pretrained model* — not training from scratch — could reach single-digit-
to-low-teens CER on real target manuscripts using as few as 15-65 real
target lines, provided the pretraining data was well matched in appearance
and language to the target. That's the more relevant comparison for your
case, and it points at something important: **148 lines fine-tuned from a
well-matched base can go a lot further than 148 lines fine-tuned from a
poorly-matched one.** The gap between your 71-80% CER and that study's
12-25% CER at similar line counts is far more likely explained by
base-model match than by raw line count. This directly bears on §2 below.

**Evaluation volume:** the same study's test sets ranged from 65 lines (its
smallest, single-author Washington dataset) up to 700+ lines, and its
validation sets from 65 to 976 lines. Even the smallest benchmark in active
research use held out over 3x more test lines than your current 19-line
c2av12 holdout, and none of them evaluated on a single page/plate the way
your holdout currently does. **Concretely: aim for at least 60-100+ held-out
lines for both val and test, ideally spread across 3+ plates each** rather
than one plate apiece — this is what turns a CER number into something you
can trust rather than a single noisy point estimate. With only 12 plates
total, the practical way to get there without new annotation is k-fold /
leave-one-plate-out CV (§4 below), which reuses the same 178 lines
differently rather than needing more of them.

## Three (plus one) ways to lower CER, given what §1 ruled out

### 1. Fix the measurement before trusting any more model deltas

Run leave-one-plate-out CV across all 12 plates using the exact v3 0.2
recipe (no freeze-rows). This gives a CER *distribution* across 12 folds
instead of one point estimate from one plate. Two direct payoffs:
- Tells you whether 71.43% vs 80.09% is a real regression or within the
  natural plate-to-plate variance (if the 12-fold spread is ±10 points,
  §1's "failure" is inconclusive; if it's ±2-3 points, it's a real result).
- Gives you a legitimate answer to "what's our CER" that isn't hostage to
  which single plate got held out.
This costs ~12 GPU-hours at the timing you already have, no new annotation,
and every other lever below should be judged against this distribution,
not a single-plate number.

### 2. Insert an intermediate domain-adaptation stage before the Sogdian delta (highest-leverage, literature-backed)

§1's falsification is actually informative: if freezing old-class rows
didn't help, the shared-class errors aren't a classifier-drift problem,
they're a **feature-level mismatch** between whatever Sophro Mhiro was
originally trained on and the c2av manuscript's actual hand/degradation
profile. The row-freeze couldn't fix that because it never touched
features, only classifier weights.

The fix suggested by the literature above is to change *what the backbone
has adapted to* before the tiny Sogdian delta is layered on, rather than
trying to force a 148-line dataset to do double duty (teach 5 new classes
*and* correct backbone mismatch at once):

1. Fine-tune Sophro Mhiro's backbone on the **Vienna 2024 Syriac Ground
   Truth dataset** you already used for the Payne Smith project — real
   historical Syriac manuscript material, almost certainly a closer visual
   and paleographic match to Turfan-era Syriac-script writing than whatever
   Sophro's original training distribution was. This dataset is large
   enough for a genuine full fine-tune (order of magnitude above the
   ~230-line convergence floor), so this step doesn't carry the same
   catastrophic-forgetting risk that unfreezing on 148 c2av lines did —
   there's enough real signal for the backbone to adapt safely.
2. Take *that* adapted model (call it `sophro_syriac_adapted`) as the new
   base, then apply the existing recipe on top: `--resize union
   --freeze-backbone 999999` fine-tuned on the 148 c2av lines for the 5
   Sogdian classes, exactly as in v3's 0.2.
3. Re-run the same 3-plate triangulation (and the LOOCV from #1) to compare
   against the 71.43% baseline.

This is the one lever in this whole arc with a concrete literature citation
showing it can matter more than architecture tweaks: pretraining-data
match (appearance, language, script era) drove larger CER swings in
published few-shot HTR results than any of the freeze/unfreeze schedule
choices tried so far in your own experiments. It's also the most direct
answer to what §1 actually falsified — a feature-level problem needs a
feature-level (backbone-adaptation) fix, done on data with enough volume
to be safe, not on the 148 lines that are also trying to teach 5 brand-new
classes.

### 3. Heavier augmentation + synthetic data (still valid, now secondary)

Keep §3/§2 from the v4 plan (heavier `ocropy_degrade`/`distort_line`
augmentation of the real 148 lines, and `ketos linegen` synthetic lines
once font glyph coverage for the 3 Sogdian-specific codepoints is
confirmed). These remain useful, but the evidence above suggests they're
a smaller lever than #2 — augmentation and synthetic data expand
*visual diversity* around the same underlying feature space, whereas #2
changes what the feature space actually represents. Run #3 after or
alongside #2, not instead of it.

### 4. More real annotation, but targeted rather than volume-maximizing

Given the literature above, the annotation target isn't "more lines in
general" — it's specifically: (a) a handful more *plates* (not just more
lines from the same 2-3 plates) so val/test can be spread across 3+ plates
each per Q2's guidance, and (b) continued prioritization of lines
containing rukkakha/zhain/fe, which are still the rarest classes regardless
of which backbone they're layered on top of. This is the one lever with no
technical substitute, but it's more effective paired with #2 than alone —
per the cited study, the ceiling a fixed amount of real fine-tuning data
can reach is set largely by how well-matched the starting model already is.

## Suggested order

1. **§1 LOOCV** — cheap, decisive, tells you whether to trust the 80.09%
   regression at all.
2. **§2 Vienna-adapted intermediate backbone** — the new, literature-backed
   lever; start this in parallel with #1 since it doesn't depend on the
   LOOCV result.
3. **§3 augmentation/synthetic** — layer on top of whichever backbone (base
   Sophro or Vienna-adapted) wins after #1/#2.
4. **§4 targeted annotation** — ongoing, highest ceiling long-term, pursue
   in parallel with the above as scholar time allows.

## What §1 ruled out (carried forward from the v4 result)

- Freezing old-class classifier rows: does not help, holdout got worse.
- The "old-row weight drift" hypothesis: falsified. The 87.3%-shared-class-
  error finding from 0.1 is more likely feature/backbone-level mismatch,
  which is what §2 above targets directly for the first time.
