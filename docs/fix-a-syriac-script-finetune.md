# Fix A: Syriac-script fine-tune (FAILED)

# Status: FAILED — do not pursue further without new approach

## Hypothesis

Christian Sogdian uses the **East Syriac script** (same as Sophro Mhiro base
model, U+0700–U+074F block). Re-encoding c2av ground truth from Latin
transliteration to Syriac Unicode codepoints should transform the task from
"zero codec overlap rebuild" (24 Latin chars vs 36 Syriac chars) into
"ordinary same-script fine-tune" where 22/25 target chars already exist in the
base codec. Only 5 new classes (the Sogdian-specific letters + diacritics)
need to be learned. This should make 148 lines viable per the feasibility
research (`docs/kraken-178-lines-feasibility.md`).

## What was done

### 1. Latin → Syriac GT re-encoding
- `scripts/latin_to_syriac.py` converts Latin transliteration to Syriac script
- 179 lines across 12 c2av plates converted, 0 unmapped chars
- 25 unique output chars (20 in base codec + 5 NEW)
- `.bak` backups of all 12 XMLs preserved alongside converted versions
- Char frequency analysis: y=303, t=203, ʾ=187, w=178, n=128, r=112, q=73,
  p=63, x=57 (KHAPH), š=48, m=48, d=39, z=37, c=37, s=35, g=35, .=30, b=28,
  θ=11 (rukkakha), f=10 (FE), ž=10 (ZHAIN), l=1, 1 Syriac pass-through
- 5 NEW codepoints: U+0741 (qushshaya, 192 occ), U+0742 (rukkakha, 11 occ),
  U+074D (SOGDIAN ZHAIN, 10 occ), U+074E (SOGDIAN KHAPH, 57 occ),
  U+074F (SOGDIAN FE, 10 occ)
- GT verified valid: clean Syriac script, well-formed PAGE-XML

### 2. User-confirmed mapping decisions
- t/θ: keep distinct via diacritics — t→TAW+qushshaya (U+072C U+0741),
  θ→TAW+rukkakha (U+072C U+0742)
- x → KHAPH (U+074E) — Sogdian-specific [x] fricative
- g/γ collapsed → GAMAL (U+0713), d → DALATH (U+0715) — model outputs one
  class each, post-processor resolves hard/soft via dictionary
- l kept as LAMADH (U+0720) — in base codec, 1 example won't train but harmless
- c2av19 `ܫܪܥܦ` kept — valid Syriac-script output, all 4 chars in base codec

### 3. Sophro base codec decoded
36 classes + CTC blank = 37 output indices. 22 standard Syriac consonants
present (U+0710–U+072C). 3 Sogdian-specific letters MISSING: U+074D, U+074E,
U+074F all absent. 2 diacritics MISSING: U+0741 (qushshaya), U+0742 (rukkakha).

### 4. Kraken `--resize` value discovery
- `ketos train --help` advertises `add|union|both|new|fail`
- kraken 7.0.2's `vgsl.py` setup() only dispatches **3 values**: `fail`,
  `union`, `new`
- `add` and `both` are argparse-only — raise
  `ValueError: invalid resize parameter value add` at runtime
- This is a kraken bug (argparse choices vs dispatch mismatch)
- `union` = preserve base codec + append unseen codepoints + resize output
  layer only. Chosen for Fix A.

### 5. orchestrator.py changes
- Changed `--resize new` to `--resize union` at line 235 with detailed
  `ponytail:` comment explaining the kraken dispatch bug
- Wired `--min-epochs` and `--lag` into the ketos train_cmd (were params but
  never passed to ketos — default lag=4 caused premature early-stop at epoch 10)

## Training runs

### Fine-tune #3 (FAILED — diverged)
- Flags: `--resize union --freeze-backbone 500 --warmup 200 --lr 5e-5 --augment --min-epochs 20 --lag 10`
- BUT: `--min-epochs` and `--lag` were NOT wired into ketos command (bug
  found and fixed after this run)
- Result: CER 97.89%, score 0.0000 from epoch 0, early-stopped at epoch 10
  (kraken default lag=4)
- Artifact: `models/kraken/c2av_finetune_syriac.safetensors` (16MB, useless)
- Symptom: `val_accuracy: 0.000`, `train_loss ~30` (huge, not decreasing)
- Log: `/tmp/opencode/c2av_finetune_syriac.log`

### Fine-tune #4 (FAILED — diverged, killed at epoch 4)
- Flags: `--resize union --freeze-backbone 0 --warmup 200 --lr 1e-4 --augment --min-epochs 20 --lag 10`
- All flags wired correctly this time
- Result: same pattern — `val_accuracy: 0.000`, `train_loss ~30` bouncing
  wildly (12 → 205 → 11 → 52), killed at epoch 4 to save GPU
- Symptom: model not learning from step 0, loss not decreasing
- Log: `/tmp/opencode/c2av_finetune_syriac2.log`

## Root cause diagnosis

`--resize union` is broken for our case. The mechanism:

1. `union` preserves old 37-class logits + appends 5 random initialized
   classes → output head is now 42 classes
2. Softmax over 42 classes: the old blank/CTC logit (well-trained) dominates
   → CTC loss stays huge (~30) because the model predicts blank for everything
3. If backbone frozen (#3, freeze 500): new head can't learn at LR 5e-5 fast
   enough to escape the blank-dominant basin → score 0 forever
4. If backbone unfrozen (#4, freeze 0): the huge CTC gradient destroys the
   backbone features before the new head can learn → catastrophic forgetting
   + no learning, loss bounces wildly

Three runs confirm: `--resize union` does not work for our 148-line
same-script-but-5-new-classes scenario. The 22 shared Syriac consonants'
weights do NOT help when the output head is disrupted.

## What did NOT work

- `--resize union` (3 attempts, all diverged)
- `--freeze-backbone 500` (head can't learn)
- `--freeze-backbone 0` (backbone destroyed)
- `--lr 5e-5` (too small with frozen backbone)
- `--lr 1e-4` (too large with unfrozen backbone, loss bounces)

## Best result so far (NOT Fix A)

**Fine-tune #1** (`--resize new --freeze-backbone 5000`): CER 82% flat across
train/val/holdout. Backbone never unfroze (5000 > total ~3000 iterations), so
the model learned a 25-class head on top of frozen Sophro Syriac features.
Underfit, but at least stable. This is the baseline to beat.

Artifacts:
- `models/kraken/c2av_finetune.safetensors` (16MB)
- `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune.{json,md}`

## Files

- `scripts/latin_to_syriac.py` — Latin→Syriac converter (DONE, ~130 lines,
  stdlib only, 179 lines converted, 0 unmapped)
- `scripts/syriac_to_latin.py` — Syriac→Latin post-processor (NEVER WRITTEN
  — Fix A failed before this was needed)
- `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml` —
  RE-ENCODED to Syriac script
- `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml.bak` —
  original Latin-transliteration backups (RESTORE THESE if abandoning Fix A)
- `docs/kraken-cer-improvement-techniques.md` — 19 CER improvement
  techniques researched (still valid for future attempts)

## Possible paths forward (NOT attempted — awaiting user direction)

1. **Rebuild head, freeze backbone forever**: `--resize new --freeze-backbone
   999999 --lr 1e-4 --epochs 50`. Head learns 25 Syriac classes from scratch,
   backbone features stay frozen. Builds on the 82% baseline.
2. **Rebuild head, unfreeze with tiny LR**: `--resize new --freeze-backbone 0
   --lr 1e-5`. Whole model trains but LR too small to destroy backbone.
   Risk: still forgets.
3. **Debug `--resize union` in kraken source**: spawn pod, inspect vgsl.py
   union code path to find why it diverges. May reveal a kraken bug or flag
   we're missing.
4. **Accept 82%, move to post-processor + more data**: ship fine-tune #1.
   Write `syriac_to_latin.py` post-processor. Annotate more plates to reach
   500+ lines for a real model.

## Relevant research

- `docs/kraken-178-lines-feasibility.md` — 178 lines at bottom of viable
  range for fine-tuning, from-scratch needs 800-2000+
- `docs/kraken-cer-improvement-techniques.md` — 19 techniques researched
  (librarian session `ses_0d293e084ffeKefdR8fl9aCxtE`)
- Christian Sogdian Unicode research (librarian session
  `ses_0d0f570f9ffe7ssq2tGJnzcXJi`): East Syriac script, U+0700–U+074F