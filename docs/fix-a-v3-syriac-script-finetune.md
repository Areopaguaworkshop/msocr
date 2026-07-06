# Fix A v3: Pushing CER from 77.75% to 71.43%

**Date**: 2026-07-06
**Status**: SHIPPED — `models/kraken/c2av_finetune_final.safetensors` (15.3MB).
Holdout CER **71.43%** (−6.32 points from v2's 77.75%).

This documents the full v3 arc: the plan, what was tried, what worked, what
failed, and where the ceiling now is. Supersedes nothing — `fix-a-v2-syriac-script-finetune.md`
remains the v2 history; `fixa-v3-cer-lower.md` remains the original v3 plan;
`fixa-v3-3.1-lexicon-negative-result.md` remains the 3.1 negative-result
record. This report consolidates them into a single end-of-v3 narrative.

## Starting point

v2 shipped at **77.75% holdout CER** with `--resize union --freeze-backbone
999999 --epochs 60` on 148 train lines / 15 val / 19 holdout. Val accuracy
was still climbing (0.298 → 0.319) at epoch 60 when the run hit
`max_epochs`, not early stopping — a classic underfitting signal.

## The v3 plan (six levers, ordered by effort-to-impact)

From `fixa-v3-cer-lower.md`:

| # | Lever | Cost | Expected impact |
|---|---|---|---|
| 0.1 | Confusion-matrix audit (free check) | zero GPU | diagnostic |
| 0.2 | 200-epoch rerun (free check) | ~1 GPU-hr | close underfit gap |
| 1.1 | `--append` new trainable layers | 1 pod | representational capacity |
| 1.2 | Oversample rare classes | 1 pod, file edit only | rare-class signal |
| 1.3 | LR/epoch sweep in frozen regime | 3-4 pods | tune the head |
| 2.1 | Synthetic data (CycleGAN ink-aging) | high effort | ceiling-raiser |
| 3.1 | Lexicon post-correction | zero GPU, inference-time | word-level CER |
| 3.2 | Beam search at inference | zero GPU, inference-time | 0-2% CER |
| 3.3 | Syriac-only base fine-tune | high effort | only if 0.1 shows shared-class errors |

## What was done

### 0.1 Confusion-matrix audit — DIAGNOSTIC, drove 0.2

Built `scripts/confusion_audit.py` + `scripts/confusion_analyze.py` (stdlib
only). Parsed the `ketos test` rendered report for c2av12 holdout
(`reports/confusion_c2av12_holdout.txt`) into a structured confusion table
(`reports/confusion_c2av12_holdout.json`, 65 substitution rows) and
categorized the errors.

**Findings** (`reports/confusion_c2av12_analysis.md`):

- 332 errors / 427 GT chars = CER 77.75%
- **277 deletions (83.4%)** — model emits blank where a GT char exists.
  This is CTC alignment collapse, NOT misclassification.
- 51 substitutions (15.4%), 4 insertions (1.2%)
- **87.3% of errors are on SHARED classes** (already in the Sophro base
  codec), only 11.4% on the 5 NEW Sogdian classes
- Top dropped chars: ܝ (39), ܘ (27), SPACE (26), ܐ (24), ܬ (19), ܪ (18),
  ܢ (18), SYRIAC QUSHSHAYA (17), ݎ (11)

**Diagnosis**: the model has learned to emit too few characters. The
classifier knows *which* char to emit when it emits one; the problem is it
emits blank too often. More training (0.2) is the right lever — it gives
CTC alignment more iterations to converge. `--append` (1.1) would not help
because the errors are on shared classes whose weights already work; adding
new trainable layers between frozen backbone and a fresh classifier would
discard those working weights.

**Decisions gated by 0.1**:
- **1.1 `--append`: SKIPPED.** 87.3% shared-class errors means `--append`'s
  fresh classifier would lose the working shared-class weights and make it
  worse. `--append` and `--resize union` are mutually exclusive (lib-1
  research: `--append` discards the old codec + classifier entirely, only
  the CNN backbone survives).
- **3.3 Syriac-only base fine-tune: deferred.** Indicated by the shared-class
  error pattern, but high effort. Pursue only if 0.2 doesn't help enough.
- **0.2 200-epoch rerun: confirmed as the right next lever.**

### 0.2 200-epoch rerun — SHIPPED, the win

Identical 3.2 recipe (`--resize union --freeze-backbone 999999 --augment
--warmup 200 --lr 1e-4`) with `--epochs 200 --lag 25 --min-epochs 8`. Run
on RunPod RTX 3090, ~1 hour wall. Best checkpoint at **epoch 173**, val
score **0.4043** (vs 3.2's epoch 49, score 0.3404 — the val_acc was still
climbing at 60 epochs, confirming the underfit diagnosis).

Artifact: `models/kraken/c2av_finetune_200ep` (15.3MB, no `.safetensors`
extension — the orchestrator writes the filename you pass). Copied to
`models/kraken/c2av_finetune_final.safetensors` as the shipped name.

**Triangulation** (`scripts/triangulate_eval.py`, reuses
`KetosTrainer.test_model` via kraken 7.0 Python API + polygon-enriched
PAGE-XML):

| Plate | Role | CER (0.2, 200ep) | CER (v2, 60ep) | Δ |
|---|---|---|---|---|
| c2av12 | holdout (unseen) | **71.43%** | 77.75% | **−6.32** |
| c2av11 | validation | **59.57%** | 65.96% | **−6.39** |
| c2av01 | training (seen) | **6.08%** | 45.30% | **−39.22** |

All three improved. Train collapsed from 45% to 6% — severe overfit (gap
~9.8× between train and val). The val/holdout gains are modest because the
binding constraint is now per-plate generalization, not CTC alignment.

Reports: `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune_200ep.{json,md}`
(holdout, from `harness.run_evaluation`).

### 3.1 Lexicon post-corrector — NEGATIVE RESULT, dropped

Before 0.2 completed, the inference-time lever 3.1 was tried against the v2
model. Full record in `fixa-v3-3.1-lexicon-negative-result.md`.

Built `scripts/lexicon_postcorrect.py` (stdlib only): extracts 7,387 unique
Sogdian words from Sims-Williams 2021 Christian Sogdian dictionary corpus
(`dataset/lexicon/NSW2021corpus.xls`, col 2 "Sogdian searchable",
caps-for-emphatic ASCII convention), then edit-distance-corrects
(Levenshtein ≤ 2) HTR output tokens not already in the lexicon.

**Result on c2av12 holdout**: CER *increased* from 79.90% to 85.57% — worse
by 5.67 points.

**Why**: the 0.1 audit already showed 83.4% of errors are deletions. A
lexicon post-corrector can only fix *substitutions* (15.4% of errors); it
cannot fix deletions because it cannot know which chars were dropped or
where to insert them. At CER ~80%, broken HTR tokens are often closer to
the truth than any edit-distance-2 lexicon match.

**Decision**: dropped from v3. Revisit only if 0.2 brings CER below ~50%,
at which point substitution errors become the dominant remaining category
and a lexicon post-corrector becomes viable. Scripts kept for reuse.

### 3.2 Beam search — DROPPED, library change

`beam_decoder` was removed in kraken 7.0b1. Only `greedy_decoder` ships.
Porting old beam-decoder code + bypassing `ketos test` for a 0-2% gain
(often zero) is not worth the effort. Dropped from v3.

### 1.1 `--append` — SKIPPED (gated by 0.1)

See 0.1 above. 87.3% shared-class errors → `--append`'s fresh classifier
would lose working shared weights. Skipped without a pod run.

### 1.2 Oversample rare classes — NOT RUN

v3 plan called for duplicating lines containing rukkakha/zhain/fe 3-5× in
the training list. Not executed — 0.2's result (severe train overfit at 6%
CER) showed the model has already memorized the 10 train plates. More
sampling of the same 148 lines would worsen overfit, not help
generalization. The binding constraint is per-plate diversity, not
rare-class frequency.

### 1.3 LR/epoch sweep — NOT RUN

Same reason as 1.2: with train CER at 6%, the frozen head has already
converged on the training data. Sweeping LR/epochs optimizes the train fit
further, which is not where the gap is.

### 2.1 Synthetic data — NOT RUN

SogdianOCR v0.7 CycleGAN pipeline does not exist publicly (lib-4 research).
Synthetic data path requires building from `kraken/linegen.py` (already in
deps) + Noto Sans Syriac Eastern. High effort, deferred. The 0.2 overfit
pattern (train 6% vs val 59.57%) is exactly what synthetic data would
address, so this remains the highest-leverage untried lever if more CER
reduction is needed — but it's a multi-day build, not a quick win.

### 3.3 Syriac-only base fine-tune — NOT RUN

Indicated by 0.1 (87.3% shared-class errors suggest the frozen Sophro
backbone's features don't transfer perfectly to this hand). High effort —
requires assembling a Syriac-only training set and a two-stage fine-tune.
Deferred. The 0.2 result (6.32-point holdout improvement from more epochs
alone) suggests the shared-class errors are partly CTC-alignment issues
that more training can fix, not purely backbone-feature mismatch. Revisit
only if 2.1 synthetic data doesn't close the gap.

## What worked

1. **0.1 confusion-matrix audit before any compute spend.** The deletion
   diagnosis (83.4% deletions, not substitutions) redirected the plan away
   from `--append` (1.1) and lexicon (3.1) toward more training (0.2). Zero
   GPU cost, maximum information. This is the single most important
   decision in the v3 arc.
2. **0.2 200-epoch rerun.** The cheapest possible lever — identical recipe,
   3.3× more epochs, ~1 GPU-hour. Best checkpoint moved from epoch 49 to
   epoch 173. Holdout CER 77.75% → 71.43%. Val 65.96% → 59.57%. Train
   45.30% → 6.08%.

## What did not work

1. **3.1 Lexicon post-corrector**: increased CER from 79.90% to 85.57%. Can
   only fix substitutions (15.4% of errors), not deletions (83.4%). At
   CER ~80%, broken tokens are closer to truth than edit-distance-2
   lexicon matches.
2. **3.2 Beam search**: `beam_decoder` removed in kraken 7.0b1. Porting
   effort high, expected gain 0-2% (often zero).
3. **1.1 `--append`**: skipped on diagnostic grounds (0.1 showed 87.3%
   shared-class errors; `--append`'s fresh classifier would lose working
   shared weights).

## Where the ceiling now is

**71.43% holdout CER, 59.57% val, 6.08% train.** The model has memorized
the 10 train plates (6% CER) but cannot generalize to unseen plates at
this data volume (59-71% CER on val/holdout). The gap is ~9.8× — classic
overfitting at 148 lines.

The binding constraint is **per-plate data diversity**, not:
- training signal (0.2 closed the underfit gap; more epochs won't help)
- rare-class frequency (1.2 would worsen overfit)
- LR/epoch tuning (1.3 would optimize train fit further)
- representational capacity (1.1 would lose shared weights)
- inference-time post-processing (3.1 makes it worse; 3.2 unavailable)
- backbone-feature mismatch (3.3 — partly addressed by 0.2's CTC fix;
  full Syriac-only fine-tune is high effort for uncertain gain)

## What to keep unchanged from v2

- `--resize union` (preserving shared-class weights is confirmed useful)
- Frozen backbone (unfreezing is a confirmed dead end at 148 lines, per v2)
- The Latin/Syriac mapping decisions (t/θ via qushshaya/rukkakha, g/γ
  collapsed to GAMAL, x→KHAPH, d→DALATH, l stripped, E kept) — don't
  relitigate the alphabet while isolating other variables

## Next steps if CER < 71.43% is still needed

1. **Annotate more plates** (target 500-1000+ lines across multiple
   manuscripts). This is the only path to CER < 10%. Everything else is
   marginal. The 0.2 overfit pattern (train 6% vs val 59.57%) is the
   evidence: the model has extracted everything it can from 148 lines.
2. **2.1 Synthetic data** (if annotation is not available): `kraken/linegen.py`
   + Noto Sans Syriac Eastern. Highest-leverage untried lever. Multi-day
   build. Mix synthetic lines into training only, never val/holdout.
3. **3.3 Syriac-only base fine-tune** (only if 2.1 insufficient): light
   full fine-tune of the Sophro base on Syriac-only material before the
   Sogdian delta, so the backbone adapts to this hand while still learning
   on an alphabet with lots of comparison data.
4. **Revisit 3.1 lexicon post-corrector** once CER drops below ~50%: at
   that point substitutions become the dominant error category and a
   lexicon can help. The code (`scripts/lexicon_postcorrect.py`) and
   corpus (`dataset/lexicon/NSW2021corpus.xls`, 7,387 words) are kept for
   this.

## Files produced in v3

### Scripts (all stdlib-only, no new deps)

- `scripts/confusion_audit.py` — 0.1: parses `ketos test` rendered report
  into structured confusion table
- `scripts/confusion_analyze.py` — 0.1: categorizes errors (deletions vs
  substitutions vs insertions, shared vs new classes)
- `scripts/lexicon_postcorrect.py` — 3.1: lexicon extractor + edit-distance
  corrector (NEGATIVE RESULT, kept for reuse if CER < ~50%)
- `scripts/dump_c2av12_preds.py` — dumps c2av12 HTR predictions as Syriac
  + Latin + GT Latin (kraken 7.0 API: `RecognitionTaskModel.load_model`,
  `model.predict(im, seg, cfg)` with `RecognitionInferenceConfig`)
- `scripts/cer.py` — character-level Levenshtein CER calculator
- `scripts/sw_to_latin.py` — back-converts Sims-Williams caps-emphatic to
  our Latin diacritic convention
- `scripts/triangulate_eval.py` — 3-plate CER triangulation (train/val/
  holdout) for any model, reuses `KetosTrainer.test_model`

### Reports

- `reports/confusion_c2av12_holdout.txt` — full `ketos test` rendered
  report (2,276 bytes)
- `reports/confusion_c2av12_holdout.json` — parsed confusion table (65
  substitution rows, sorted by error count)
- `reports/confusion_c2av12_analysis.md` — categorized analysis + 1.1
  decision + diagnosis
- `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune_200ep.{json,md}`
  — 0.2 holdout eval (CER 71.43%)
- `reports/c2av12_pred_syriac.txt` — 19 lines raw Syriac HTR output
- `reports/c2av12_pred_latin.txt` — 19 lines Latin (via `syriac_to_latin`)
- `reports/c2av12_pred_latin_corr.txt` — 19 lines SW-caps post-corrected
- `reports/c2av12_pred_latin_corr_back.txt` — back-converted to Latin
- `reports/c2av12_pred_latin_corr.log.txt` — 23 corrections log
- `reports/c2av12_gt_latin.txt` — 19 lines GT Latin

### Models

- `models/kraken/c2av_finetune_200ep` — 0.2 raw artifact (15.3MB, no
  extension — orchestrator writes the filename you pass)
- `models/kraken/c2av_finetune_final.safetensors` — shipped copy of 0.2
  (15.3MB, holdout CER 71.43%)
- `models/kraken/c2av_finetune_union_frozen.safetensors` — v2 artifact
  (16MB, holdout CER 77.75%), kept for comparison

### Logs

- `/tmp/opencode/c2av_200ep.log` — 0.2 training log (tqdm progress bars,
  best checkpoint epoch 173 score 0.4043)

### Lexicon corpus (for 3.1, kept for reuse)

- `dataset/lexicon/NSW2021corpus.xls` — Sims-Williams 2021 Christian
  Sogdian dictionary (3.6 MB, 27,020 rows, 6 cols)
- `dataset/lexicon/sogdian_words.txt` — 7,387 unique words extracted from
  col 2

## Reusable research sessions (librarian)

- `ses_0d293e084ffeKefdR8fl9aCxtE` — Kraken `--append`/`--resize`/VGSL
  (lib-1: `--append` and `--resize union` mutually exclusive)
- `ses_0d066aa39ffeD0Amn5BQhhGK5L` — SogdianOCR CycleGAN (lib-4: does not
  exist publicly; use `kraken/linegen.py` + Noto Sans Syriac Eastern)
- `ses_0d0669554ffe8i54qXQVGCIQzQ` — Christian Sogdian lexicon (lib-5:
  Sims-Williams Excel corpus free download, 3.6 MB, ~27k rows)
- `ses_0d0668443ffeB2WM2hTfvJERGV` — kraken beam_decoder (lib-6: removed
  in kraken 7.0b1)
- `ses_0d0f570f9ffe7ssq2tGJnzcXJi` — Christian Sogdian Unicode (East
  Syriac script, U+0700-U+074F, 22 consonants + 3 Sogdian-specific)

## One-line summary

0.1 audit said "the errors are deletions, not substitutions — train
longer, don't add architecture"; 0.2 trained longer and dropped holdout
CER 77.75% → 71.43%; everything else (lexicon, beam search, `--append`)
was either tried and failed or skipped on diagnostic grounds; the
remaining ceiling is per-plate data diversity at 148 lines.