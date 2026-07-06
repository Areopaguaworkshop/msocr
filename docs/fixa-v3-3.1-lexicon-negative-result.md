# 3.1 Lexicon Post-Corrector — Negative Result

**Date**: 2026-07-06
**Status**: FAILED — post-corrector increases CER. Do not ship.

## What was built

- `scripts/lexicon_postcorrect.py` — extracts 7,387 unique Sogdian words from
  Sims-Williams 2021 Christian Sogdian dictionary corpus
  (`dataset/lexicon/NSW2021corpus.xls`, col 2 "Sogdian searchable"), then
  edit-distance-corrects (Levenshtein ≤ 2) HTR output tokens not already in
  the lexicon.
- `scripts/dump_c2av12_preds.py` — dumps model predictions on c2av12 holdout
  as Syriac + Latin (via `syriac_to_latin.py`), plus GT Latin.
- `scripts/cer.py` — character-level Levenshtein CER between pred and GT.
- `scripts/sw_to_latin.py` — back-converts Sims-Williams caps-emphatic to our
  Latin diacritic convention (for measurement only).

## Measurement (c2av12 holdout, 19 lines, 388 GT chars)

| Stage | CER |
|---|---|
| Raw HTR → `syriac_to_latin` | **79.90%** (310/388) |
| + lexicon post-correct (SW space, no back-convert) | 90.21% (350/388) — apples-to-oranges |
| + lexicon post-correct (back-converted to Latin) | 85.57% (332/388) |

**Lexicon post-corrector increases CER by 5.67 points.** Do not ship.

## Why it fails

The 0.1 confusion-matrix audit (`reports/confusion_c2av12_analysis.md`) already
diagnosed this:

- **83.4% of errors are deletions** (model emits blank instead of char — CTC
  alignment collapse).
- Only **15.4% are substitutions**.

A lexicon post-corrector can only fix *substitutions* (wrong char → right char
within edit distance 2). It cannot fix *deletions* because it cannot know
*which* chars were dropped or *where* to insert them. With CER ~80%, the
average token has 1-2 missing chars, so the broken HTR token is often *closer
to the truth* than any edit-distance-2 lexicon match.

Concrete examples from the log (`reports/c2av12_pred_latin_corr.log.txt`):
- `WYR→ZY` (d=2): original `wyr` vs GT `qwrθ yty xw` — lexicon match is worse.
- `PTA→JnA` (d=2): original `ptʾ` vs GT `pwrn c` — lexicon match is worse.
- `TSA→TwA` (d=1): original `tsʾ` vs GT `pwryc yʾ` — both wrong, lexicon
  doesn't help.

## What would help

1. **0.2 200-epoch rerun** (highest confidence): the 3.2 union-frozen run was
   still improving val_acc at epoch 60 when it early-stopped. More epochs may
   improve CTC alignment and reduce the deletion rate. Once CER drops below
   ~50%, a lexicon post-corrector becomes viable.
2. **Language model rescoring at CTC decode time** (not post-hoc): beam search
   with a char-level LM would explicitly trade off emission vs blank
   probabilities. But `beam_decoder` was removed in kraken 7.0b1 (lib-6
   research), so this requires porting old code — dropped from v3 plan.
3. **Lexicon with insertion-aware matching**: instead of Levenshtein, match
   HTR token to lexicon word by *insertion-only* distance (how many chars
   must be added to the HTR token to reach the lexicon word, preserving
   order). This directly addresses the deletion-error profile. Not built —
   the 0.2 rerun is a higher-leverage lever and may make it unnecessary.

## Decision

**Drop 3.1 from v3.** Proceed to **0.2 200-epoch rerun** as the next lever.
If 0.2 brings CER below ~50%, revisit 3.1 with the same code.

## Files

- `scripts/lexicon_postcorrect.py` — kept for potential reuse after 0.2
- `scripts/dump_c2av12_preds.py` — kept, useful for any future pred inspection
- `scripts/cer.py` — kept, generally useful
- `scripts/sw_to_latin.py` — kept, measurement utility
- `dataset/lexicon/NSW2021corpus.xls` — Sims-Williams corpus (3.6 MB)
- `dataset/lexicon/sogdian_words.txt` — 7,387 unique words extracted
- `reports/c2av12_pred_syriac.txt` — 19 lines raw Syriac HTR output
- `reports/c2av12_pred_latin.txt` — 19 lines Latin (via syriac_to_latin)
- `reports/c2av12_pred_latin_corr.txt` — 19 lines SW-caps post-corrected
- `reports/c2av12_pred_latin_corr_back.txt` — back-converted to Latin
- `reports/c2av12_pred_latin_corr.log.txt` — 23 corrections log
- `reports/c2av12_gt_latin.txt` — 19 lines GT Latin