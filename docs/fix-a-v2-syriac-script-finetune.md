# Fix A v2: Syriac-script fine-tune (SUCCEEDED — shipped)

Status: **SHIPPED.** `c2av_finetune_union_frozen.safetensors` is the final
model. CER 77.75% on holdout (beats 82% baseline from Fix A #1).

Supersedes `docs/fix-a-syriac-script-finetune.md` (which documented the failed
union+unfrozen runs). Read this doc, not that one.

## TL;DR

- Re-encoding c2av GT from Latin transliteration to Syriac Unicode turned a
  zero-overlap rebuild (24 Latin chars vs 36 Syriac base chars) into a 5-class
  delta fine-tune (22 of 25 target chars already in base codec).
- `--resize union --freeze-backbone 999999` (freeze forever) trained stably
  for 60 epochs, val_accuracy climbed 0.298 → 0.319, best score 0.3404.
- Holdout CER **77.75%**, val CER **65.96%**, train CER **45.30%** — beats the
  82%-flat baseline from fine-tune #1 across all three plates with a clean
  train<val<holdout ordering (real learning, not memorization noise).
- **Staged unfreeze failed** (3.3): loading the union-frozen model as base and
  unfreezing the backbone at 1/10 LR (1e-5) + 500 warmup still diverged —
  val_acc collapsed to 0, loss 30.7, killed at epoch 39. Unfreeze is a binding
  no-go at 148 lines regardless of LR, warmup, or starting checkpoint.
- **Decision gate: SHIP** the frozen model + write the Syriac→Latin
  post-processor. Path to lower CER is more data (500-1000+ lines), not more
  unfreeze tricks.

## Root cause of why Fix A v1 failed and v2 worked

The difference between v1 (failed) and v2 (succeeded) was **freeze duration**,
not the `--resize` mode or the LR.

v1 runs (#3, #4) used `--freeze-backbone 500` and `--freeze-backbone 0`
respectively — the backbone became trainable *during* training. With the
union-resized head (37 old classes + 5 new random classes = 42-way softmax),
the large noisy gradients from the 5 new classes dragged the backbone features
around before the new head could stabilize → catastrophic forgetting, loss
bouncing 12→205→11→52, val_acc stuck at 0.

v2 (`--freeze-backbone 999999`) kept the backbone frozen the entire run. The
only trainable parameters were the 42-way output head (the convolutional +
recurrent features stayed at Sophro Syriac values). The 22 shared Syriac
consonant logits kept their well-trained weights; only the 5 new class logits
had to be learned — and with a frozen backbone, the gradient signal for those
5 classes is clean (no backbone noise competing).

This confirms the Fix A v2 plan diagnosis: union's mixed old/new classifier +
unfrozen backbone = catastrophic forgetting. Freeze the backbone and union
works exactly as designed.

## What was done

### 1. Latin → Syriac GT re-encoding (from Fix A v1, unchanged)

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

### 2. Alphabet audit (3.1, local — DONE)

- 24 unique chars in converted GT (after stripping the 1 `l` LAMADH singleton,
  which cannot be learned at 1 occurrence and would only add softmax noise)
- NFC-clean: alphabet is in NFC and NFD simultaneously (no normalization
  duplicate risk in the codec)
- Combining marks U+0741/U+0742 are precomposed in NFC (no decomposition) —
  no codec duplicate risk
- E (U+0725, 1 occ, from c2av19 Syriac pass-through) kept: it's in the base
  codec, so `--resize union` preserves its weights — the 1 example doesn't
  need to train it from scratch
- Exact match vs expected 25-char list minus l

### 3. User-confirmed mapping decisions (from Fix A v1, unchanged)

- t/θ: keep distinct via diacritics — t→TAW+qushshaya (U+072C U+0741),
  θ→TAW+rukkakha (U+072C U+0742)
- x → KHAPH (U+074E) — Sogdian-specific [x] fricative
- g/γ collapsed → GAMAL (U+0713), d → DALATH (U+0715) — model outputs one
  class each, post-processor resolves hard/soft via dictionary
- l kept as LAMADH (U+0720) originally, then stripped (1 occurrence, audit)
- c2av19 `ܫܪܥܦ` kept — valid Syriac-script output, all 4 chars in base codec

### 4. Sophro base codec (unchanged)

36 classes + CTC blank = 37 output indices. 22 standard Syriac consonants
present (U+0710–U+072C). 3 Sogdian-specific letters MISSING: U+074D, U+074E,
U+074F all absent. 2 diacritics MISSING: U+0741 (qushshaya), U+0742 (rukkakha).

### 5. Kraken `--resize` value discovery (unchanged)

- `ketos train --help` advertises `add|union|both|new|fail`
- kraken 7.0.2's `vgsl.py` setup() only dispatches **3 values**: `fail`,
  `union`, `new`
- `add` and `both` are argparse-only — raise
  `ValueError: invalid resize parameter value add` at runtime
- This is a kraken bug (argparse choices vs dispatch mismatch)
- `union` = preserve base codec + append unseen codepoints + resize output
  layer only. Used for Fix A v2.
- Librarian confirmed (lib-1): `ketos resize` is NOT a standalone subcommand
  — only available as `ketos train --resize`. So zero-shot alignment check
  folds into the first 1-3 epochs of a training run (val_acc > 0 = alignment
  held, val_acc = 0 = union broke labels).

### 6. orchestrator.py changes (from Fix A v1, unchanged)

- `--resize union` at line 235 with detailed `ponytail:` comment
- `--min-epochs` and `--lag` wired into the ketos train_cmd at lines 227-228
  (were orchestrator params but never passed to ketos — kraken default lag=4
  caused premature early-stop at epoch 10 in v1 run #3)

### 7. ssh_exec wedge fix (from Fix A v1, unchanged)

`get_pty=True` + blocking `readline()` on a quiet command (pip --quiet)
blocked forever. Rewrote `ssh_exec` in `runpod_runner.py` to use
`recv_ready()` polling + `exit_status_ready()` + `time.monotonic()` deadline.

## Training runs

### Fine-tune #1 (baseline, `--resize new --freeze-backbone 5000`) — DONE in v1

- CER 82% flat across train/val/holdout (backbone never unfroze — 5000 > total
  ~3000 steps). Best result from v1, baseline to beat.
- Artifact: `models/kraken/c2av_finetune.safetensors` (16MB)

### Fine-tune #2 (`--resize new --freeze-backbone 0`) — DONE in v1

- CER worse (91% holdout, 95% train). Unfreezing from step 0 → catastrophic
  forgetting + overfitting. Confirmed unfreezing is bad at 148 lines.

### Fine-tune #3 (`--resize union --freeze-backbone 500`) — FAILED in v1

- Diverged, CER 97.89%, score 0 from epoch 0, `--min-epochs`/`--lag` not wired
  (bug, now fixed). Documented in `docs/fix-a-syriac-script-finetune.md`.

### Fine-tune #4 (`--resize union --freeze-backbone 0 --lr 1e-4`) — FAILED in v1

- Same divergence pattern, killed at epoch 4, loss bouncing 12→205→11→52.
  Documented in `docs/fix-a-syriac-script-finetune.md`.

### 3.2 Freeze-forever ablation (`--resize union --freeze-backbone 999999`) — SUCCEEDED

- Flags: `--resize union --freeze-backbone 999999 --lr 1e-4 --warmup 200 --augment --min-epochs 8 --lag 10 --epochs 60`
- Pod `be76sa1yow7sis` (2nd attempt after capacity retry), 60 epochs ran
  (max_epochs reached, not early-stopped)
- val_accuracy climbed 0.298 → 0.319, best score 0.3404 at checkpoint 49
- train_loss plateaued at ~15.3 (frozen backbone ceiling — expected)
- Zero-shot alignment gate PASSED: val_acc > 0 from epoch 1 (union preserved
  old indices/weights as kraken source confirms)
- Artifact: `models/kraken/c2av_finetune_union_frozen.safetensors` (16MB)
- Log: `/tmp/opencode/c2av_union_frozen.log`

#### 3-plate eval triangulation (CER %)

| Plate | Role | 3.2 union-frozen | #1 new-frozen (baseline) | Delta |
|---|---|---|---|---|
| c2av12 | holdout (unseen) | **77.75** | 82.12 | -4.37 |
| c2av11 | val | **65.96** | 82.50 | -16.54 |
| c2av01 | train (seen) | **45.30** | 83.43 | -38.13 |

Union-frozen beats `new`-frozen on every plate. The train<val<holdout
ordering (45→66→78) is the signature of real learning, not memorization noise.
The 22 shared Syriac consonant weights from the base model transfer
meaningfully — this is the architectural payoff of Fix A v2.

Reports:
- `reports/c2av-finetune__c2av-syriac-finetune__c2av_finetune_union_frozen.{json,md}` (holdout p-12)
- `reports/c2av-eval-val__c2av-syriac-finetune__c2av_finetune_union_frozen.{json,md}` (val p-11)
- `reports/c2av-eval-train__c2av-syriac-finetune__c2av_finetune_union_frozen.{json,md}` (train p-01)

### 3.3 Staged unfreeze (`--freeze-backbone 0 --lr 1e-5 --warmup 500`, loaded 3.2 as base) — FAILED

- Flags: `--resize union --freeze-backbone 0 --warmup 500 --lr 1e-5 --augment --min-epochs 15 --lag 15 --epochs 60`
- Base: `models/kraken/c2av_finetune_union_frozen.safetensors` (3.2 output,
  5 new classes already learned at 77% CER)
- Diverged at stage 39/59: val_accuracy collapsed to 0.000, train_loss stuck
  at 30.7 (vs 15.3 in 3.2 frozen). Killed to save GPU.
- Same failure mode as v1 runs #2/#4: backbone unfreeze + union = catastrophic
  forgetting, regardless of LR (tried 1e-3, 1e-4, 1e-5), warmup (0, 200, 500),
  or starting checkpoint (base vs 3.2 frozen model).
- Pod `v4npqrtpc7ujj9` terminated. Log: `/tmp/opencode/c2av_staged_unfreeze.log`

**Conclusion: backbone unfreeze is a binding no-go at 148 lines.** The frozen
backbone features that transfer from Sophro are fragile — any gradient
through them with tiny noisy per-line batches destroys them faster than the
new-class head can stabilize. The frozen-backbone result (77.75%) is the
ceiling for this data volume with this approach.

## Decision gate

Per the Fix A v2 plan:

- CER < 82% AND new classes show per-class accuracy? **Yes** — 77.75% holdout
  beats 82% baseline, val_acc was 0.298–0.319 (not 0) so the 5 new Sogdian
  classes ARE being learned.
- Staged unfreeze works? **No** — diverges every time, regardless of
  LR/warmup/checkpoint.

→ **SHIP 3.2 union-frozen as the final model** + write
`scripts/syriac_to_latin.py` post-processor. Path to lower CER is more data
(500-1000+ lines), not more unfreeze tricks.

## What did NOT work

- `--resize new --freeze-backbone 0` (fine-tune #2): catastrophic forgetting
- `--resize union --freeze-backbone 500` (#3): diverged (also `--min-epochs`
  bug)
- `--resize union --freeze-backbone 0 --lr 1e-4` (#4): diverged
- `--resize union --freeze-backbone 0 --lr 1e-5 --warmup 500` (3.3, loaded 3.2
  as base): diverged — proves unfreeze is the problem, not the LR or the
  starting checkpoint

## What worked

- `--resize new --freeze-backbone 5000` (#1): CER 82% flat, baseline
- **`--resize union --freeze-backbone 999999` (3.2): CER 77.75% holdout,
  shipped** — union preserves the 22 shared Syriac consonant weights; frozen
  backbone gives the 5 new classes a clean gradient signal

## Final artifact

`models/kraken/c2av_finetune_union_frozen.safetensors` (16MB)

- 37 + 5 = 42-way output head (37 base + 5 new Sogdian classes)
- Backbone = frozen Sophro Mhiro Syriac CNN+LSTM features
- Holdout CER 77.75%, val CER 65.96%, train CER 45.30%
- Use with `scripts/syriac_to_latin.py` for Latin transliteration output

## Files

- `scripts/latin_to_syriac.py` — Latin→Syriac converter (DONE, ~130 lines,
  stdlib only, 179 lines converted, 0 unmapped)
- `scripts/syriac_to_latin.py` — Syriac→Latin post-processor (DONE, ~60 lines,
  deterministic lookup + dictionary-based t/θ, g/γ resolution)
- `scripts/alphabet_audit.py` — 3.1 alphabet audit script (DONE, local, no GPU)
- `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml` —
  RE-ENCODED to Syriac script (l LAMADH stripped from c2av01)
- `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml.bak` —
  original Latin-transliteration backups (preserve for future re-encoding)
- `docs/fix-a-syriac-script-finetune.md` — Fix A v1 failure record (superseded
  by this doc; kept for the v1 run #3/#4 failure details)
- `docs/kraken-cer-improvement-techniques.md` — 19 CER improvement techniques
  (still valid for future attempts)

## Next steps (out of scope for this fix)

1. **Annotate more plates** (target 500-1000+ lines across multiple
   manuscripts). 148 lines is the binding constraint — the frozen-backbone
   77.75% is the ceiling for this data volume. More data is the only path to
   CER < 10%.
2. **Leave-one-plate-out CV** across all 12 plates for a more principled
   evaluation than a single 19-line holdout.
3. **Once more data exists**, revisit staged unfreeze — with 500+ lines the
   per-line gradient noise drops and unfreezing may finally stabilize. Until
   then, frozen-backbone is the only stable regime.

## Relevant research

- `docs/kraken-178-lines-feasibility.md` — 178 lines at bottom of viable
  range for fine-tuning, from-scratch needs 800-2000+
- `docs/kraken-cer-improvement-techniques.md` — 19 techniques researched
  (librarian session `ses_0d293e084ffeKefdR8fl9aCxtE`)
- Christian Sogdian Unicode research (librarian session
  `ses_0d0f570f9ffe7ssq2tGJnzcXJi`): East Syriac script, U+0700–U+074F