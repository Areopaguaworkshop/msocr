# Fix A v2: Root Cause Analysis + Implementation Plan

Status: supersedes `docs/fix-a-syriac-script-finetune.md`. Read that file
first for full history before executing anything here.

## 1. Confirmed root cause

Kraken's `add`/`union` resize path is documented to preserve every existing
output row and only append rows for new codepoints — it is not a random
full-reinit of the classifier. So the original hypothesis ("union nukes the
old weights") is not quite right. The actual failure is a **freeze-duration
mismatch**, and the three runs already on record demonstrate this cleanly:

| Run | Resize | Freeze | Result |
|---|---|---|---|
| #1 | `new` (full reinit head) | 5000 (> total ~3000 steps, effectively never unfreezes) | CER 82%, stable, underfit |
| #3 | `union` (preserve 37 + 5 new) | 500 (unfreezes early, most of training happens with backbone trainable) | diverged, val_acc 0.000, loss ~30 |
| #4 | `union` | 0 (backbone trainable from step 0) | diverged worse, loss bounced 12→205→11→52 |

The pattern: **the only stable run was the only run where the backbone
stayed frozen for the entire training window.** Every run where the backbone
became trainable — with `union`'s mixed old/new classifier — diverged.

Mechanism:
- With `union`, the 5 new rows start near-zero/random while the 37 old rows
  are fully converged. Early gradients are dominated by the new rows trying
  to move fast (correct, since they're undertrained) while the old rows are
  near their optimum and mostly just need to hold steady.
- Once the backbone is unfrozen, CTC backprop pushes gradient into the
  shared conv/LSTM encoder from *both* signals simultaneously. Because the
  new-row gradients are large and noisy (5 classes learning from a combined
  ~110 occurrences across 179 lines, several with single-digit counts), they
  drag the backbone's feature space in directions that make the previously
  correct 37-class old rows wrong. This is classical catastrophic forgetting,
  and it's worse than usual here because the "old" and "new" tasks share one
  linear classifier reading from one backbone — there's no adapter isolating
  the two.
- Loss bouncing between 12 and 205 across adjacent epochs (not a smooth
  decline) is consistent with this: the optimizer is oscillating between
  "backbone tuned for new classes, old classes broken" and vice versa,
  never settling into a joint optimum with only 179 lines and no LR decay
  fast enough to force convergence.

This is a training-schedule problem, not a fundamentally broken hypothesis.
`union` (preserving 22 shared letters' weights) should still beat `new`
(fully random head) if the schedule respects the freeze constraint that made
run #1 stable.

## 2. Diagnostic to run BEFORE any further training (cheap, no GPU-hours)

Before scheduling more runs, verify the resize actually preserved label
alignment correctly — this is a 5-minute check that would have caught a
much worse bug if one existed.

**Zero-shot alignment check:**
1. Load the base Sophro model.
2. Apply `--resize union` against the 179-line GT to produce the resized
   model, but do **not** train it — save immediately after resize, 0 steps.
2. Run inference with this untrained-but-resized model on a handful of
   lines that only contain the 20 shared Syriac consonants (no Sogdian-only
   letters).
3. Expected: predictions should be close to the base model's predictions on
   the same lines pre-resize (some noise from softmax renormalization over
   42 classes instead of 37 is fine; wildly different or blank-only output
   is not).

If step 3 shows garbage output even before any training, the resize
operation itself is mis-numbering labels (e.g. new classes inserted in the
middle of the label space, shifting indices for old classes) — a kraken bug,
not a training-schedule problem, and the plan below changes to a manual
`safetensors` weight-surgery approach instead of relying on `--resize`.
If step 3 shows reasonable output, proceed to Section 3.

## 3. Revised training recipe (Fix A v2)

### Step 3.1 — Alphabet audit (do this regardless of diagnostic outcome)
- Dump `gt_set.alphabet` from `ketos train --dry-run`-equivalent (or add a
  one-off debug print in orchestrator.py) and manually diff against the
  expected 25-char list.
- Confirm no NFC/NFD duplicate classes are hiding in the 25 (e.g. combining
  qushshaya represented two different ways). This alone would silently
  inflate the classifier and dilute already-thin per-class data.
- Decide now whether to keep the `l` singleton (1 occurrence) as a real
  class or fold it into a near-neighbor / mark it `unclassified` and exclude
  from loss — 1 example cannot be learned and may be adding class-count
  noise to the softmax for no benefit.

### Step 3.2 — Freeze-forever ablation with `union` (the key missing experiment)
Run the same recipe that produced the stable 82% baseline, but with
`--resize union` instead of `--resize new`:

```
ketos train \
  -i sophro_mhiro.mlmodel \
  --resize union \
  --freeze-backbone 999999 \
  --warmup 200 \
  --lr 1e-4 \
  --augment \
  --min-epochs 30 \
  --lag 10 \
  train_gt/*.xml
```

This isolates one variable: does preserving the 20 shared Syriac consonant
weights (union) beat a fully random 25-class head (new), when the backbone
is frozen in both cases? If union-frozen beats 82% CER, the shared weights
are genuinely useful and worth carrying into a staged unfreeze. If it
doesn't beat 82%, the shared Syriac weights aren't transferring meaningfully
at this data scale, and effort should go to Section 4 instead of further
schedule tuning.

### Step 3.3 — Staged/discriminative unfreeze (only if 3.2 beats baseline)
Do not unfreeze at a fixed step count. Instead:
1. Train with backbone frozen until validation loss on the 5 new classes
   plateaus (early stopping signal, not a magic number).
2. Unfreeze only the last LSTM layer + classifier (not full backbone) at
   1/10th the LR used for the head.
3. Only if that's stable, unfreeze earlier layers, at progressively lower
   LR per layer (discriminative fine-tuning), never all-at-once.
4. Use `--lr` in the 1e-6–1e-5 range for any backbone layer once unfrozen —
   1e-4 unfrozen (run #4) is too aggressive for full-backbone fine-tuning
   off a converged base model at this data scale.

### Step 3.4 — Class imbalance mitigation
- Given the frequency skew (y=303 down to l=1), consider weighting the CTC
  loss inversely by class frequency, or at minimum monitor per-class
  accuracy in validation reports rather than only aggregate CER — aggregate
  CER can look fine while the 5 new (rare) classes are still essentially
  unlearned, which is the whole point of this fine-tune.

## 4. Decision gate

After 3.2 (and 3.3 if applicable):
- **If new best CER < 82% and new classes show non-trivial per-class
  accuracy**: ship it, write `syriac_to_latin.py` post-processor, done.
- **If CER doesn't improve on 82% baseline**: the bottleneck is data volume,
  not architecture or schedule. Stop tuning `--resize`/`--freeze-backbone`
  combinations and move to targeted data collection — specifically more
  lines containing the 5 Sogdian-specific classes (current max is 192
  occurrences for qushshaya, several others under 60, `l` at 1), since
  per your own feasibility doc 179 lines is already at the bottom of the
  viable range for a same-script fine-tune, and a 5-new-class problem
  needs proportionally more per-class examples than a 0-new-class one.

## 5. What NOT to do

- Don't try more `--freeze-backbone` step-count guesses. The step-count
  approach is inherently fragile because the "right" number depends on when
  the new-class loss plateaus, which varies per run. Use the loss-plateau
  trigger in 3.3 instead.
- Don't raise LR to fix slow convergence while unfrozen — runs #3/#4 already
  show this destabilizes rather than speeds up learning at this data scale.
- Don't add more Sogdian-specific classes or change the mapping scheme
  (t/θ, g/γ collapse, etc.) mid-experiment — hold the alphabet fixed while
  isolating the freeze-schedule variable, or results across runs won't be
  comparable.
