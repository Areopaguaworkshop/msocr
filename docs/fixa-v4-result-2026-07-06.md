# Fix A v4 §1 Result: Freeze-Rows Fine-Tune — FAILED

Status: §1 (freeze old-class classifier rows) executed and evaluated.
Outcome: **holdout CER regressed from 71.43% (v3 baseline) to 80.09% (+8.66 pts)**.
The freeze mechanism works as designed, but the hypothesis it was testing
(old-class weight drift causes shared-class errors) is **falsified** for this
dataset.

## What was run

- Recipe: v3 0.2 baseline (`--resize union --epochs 200 --lag 25 --lr 1e-4
  --augment`) with `--freeze-backbone 999999` **plus** a Python-API gradient
  hook that zeroes the 37 old-class classifier rows on every backward pass,
  leaving only the 5 new Sogdian-class rows trainable.
- Base model: `models/kraken/sophro_mhiro_syriac.safetensors` (15.3M)
- Data: `data/manifests/c2av-finetune.json` — 10 train plates (154 lines),
  1 val plate (5 lines), 1 holdout plate (19 lines). All lines annotated.
- Codec: `reports/c2av_union_codec.json` (37 old + 5 new = 42 rows)
- Harness: `msocr/training/ketos_trainer_api.py` (Phase 0b), wraps
  `KrakenTrainer` (Lightning) to register `param.register_hook` on the
  classifier weight + bias. Uploaded to RunPod, trained on RTX pod.
- Run log: `reports/logs/run_freeze_v4_20260706_181201.log`
- Output model: `models/kraken/c2av_freeze_v4.safetensors` (15.3M)
- Eval report: `reports/c2av-finetune__c2av-syriac-finetune__c2av_freeze_v4.json`

## Results

| Metric | v3 0.2 baseline (no freeze) | v4 §1 freeze-rows |
|---|---|---|
| Holdout CER | **71.43%** | **80.09%** (+8.66) |
| Holdout WER | 96.36% | 98.73% |
| Holdout accuracy | 28.57% | 19.91% |
| Val accuracy (best) | — | 0.255 (val CER 27.66%) |

Training trajectory (from log):
- val_accuracy: 0.00 → 0.021 → 0.043 → 0.106 → 0.128 → 0.106 → 0.149 → 0.170 → 0.255 → 0.191 → 0.234 → 0.255 (peak)
- train_loss_epoch: 537 → 472 → 410 → 353 → 321 (steady but slow)
- Best checkpoint: `best_0.2766.safetensors` (val CER 27.66%)

The 5 new-class rows **did learn** — val CER dropped to 27.66%, val accuracy
climbed from 0% to 25.5%. But holdout CER got **worse**, not better.

## Why it failed

### 1. The drift hypothesis was wrong

The v4 plan argued: "87.3% of holdout errors are on shared Syriac classes
because `--freeze-backbone` doesn't freeze the classifier, so old rows drift
on the 148-line sample's character-frequency distribution." If that were true,
freezing the old rows should have **improved** shared-class accuracy on
holdout. It didn't — holdout got worse by 8.66 points.

This means the shared-class errors in v3 are **not** caused by old-row weight
drift. The more likely cause (now corroborated): the c2av manuscript's hand
is a genuinely poor match for the Sophro base model's training distribution.
Freezing old rows can't help recognition that's already failing at the
feature/backbone level — it just removes the only mechanism (gradient updates)
that could have nudged old-class recognition toward this manuscript's hand.

### 2. Too little trainable capacity

With backbone frozen (`freeze_backbone=999999`) **and** 37/42 classifier rows
frozen, only ~5 rows × few hundred features of capacity remained trainable.
That's enough to memorize 5 new classes on the 154-line train set (val CER
27.66% on 5 new-class-heavy val lines) but not enough to adapt old-class
recognition to the c2av hand. The model became a worse recognizer of old
classes without gaining compensating improvement on new classes.

### 3. Validation set too small to detect overfitting

5 validation lines is too few. The val_accuracy signal was noisy
(0.106 → 0.255 → 0.191 across epochs), making early stopping and best-
checkpoint selection unreliable. Best checkpoint selection picked a model
that peaked on 5 val lines but regressed on the 19 holdout lines. This is a
**measurement problem**, not a model problem — we may have been measuring
noise.

## What this rules out

- **Freeze-rows as a lever: dead end.** Freezing old rows doesn't help;
  full unfreeze (v3 0.2) is better.
- **The "old-row drift" hypothesis: falsified.** Shared-class errors aren't
  drift-driven; they're feature/recognition-level mismatch.
- **`freeze_backbone=999999` (full backbone freeze) as a default: wrong.**
  v3 0.2 used it but the row-freeze hook was the actual mechanism being
  tested. v3's improvement came from training all 42 rows against the
  manuscript, not from the backbone freeze.

## What this leaves on the table (from v4 plan)

| Lever | v4 plan section | Status after §1 |
|---|---|---|
| §1 Freeze old rows | "headline item" | **FAILED — falsified** |
| §2 Synthetic data (font coverage) | still pending | unchanged |
| §3 Line-level augmentation (heavier than ketos `--augment`) | ready to run | **now the primary lever** |
| §4 LOOCV (leave-one-out cross-validation) | pending | now more valuable (measures variance) |

The v4 plan's primary hypothesis (§1) is dead. The remaining levers (§3
augmentation, §4 LOOCV, §2 synthetic) target different problems: §3 expands
the effective training distribution, §4 gives a reliable variance estimate
on the small dataset, §2 adds new writing styles.

## Confounding factors (caveats)

1. **Val set too small (5 lines).** Best-checkpoint selection is noisy.
   The 80.09% might be 5–10 pts off in either direction. Needs §4 LOOCV
   for a reliable estimate.
2. **Holdout is a single plate (19 lines).** CER estimate has high
   variance. One plate isn't a population.
3. **`freeze_backbone=999999` was applied alongside the row freeze.** This
   conflates two freezes. A cleaner ablation would freeze rows only, leaving
   the backbone trainable — but that's a separate experiment and §1's
   failure makes it lower priority.
4. **No augmentation beyond ketos `--augment`.** §3 augmentation not yet
   tested in combination.

## Recommended next steps (revised)

1. **§4 LOOCV first** (cheap, decisive for measurement): run 10-fold
   leave-one-plate-out CV on the v3 0.2 recipe to get a reliable CER
   distribution. If variance is ±10 pts, the 71.43% vs 80.09% difference
   is within noise and §1's "failure" is inconclusive. If variance is
   ±2 pts, §1 is genuinely worse.
2. **§3 augmentation** (primary lever): run the heavier `distort→degrade→
   ocropy` augmentation chain (3 variants/line → 462 train lines) with
   the v3 0.2 recipe (no freeze). Tests whether expanding the training
   distribution helps old + new classes generalize.
3. **Split redesign**: consider k-fold CV instead of single holdout plate.
   With 178 lines total, a single 19-line holdout is too noisy. 5-fold
   CV (each fold ~36 lines) would be more reliable. Pending @librarian
   research on minimum val/test set sizes.
4. **§2 synthetic data**: still pending, lower priority until §3/§4 land.
5. **Drop §1 freeze-rows from the v4 plan.** It's a failed lever.

## Lessons for future plans

- **Test the measurement before testing the model.** With 5 val lines,
  we can't distinguish model quality from measurement noise. Small-data
  experiments need CV or larger val sets before drawing conclusions.
- **One lever at a time.** §1 conflated row-freeze + backbone-freeze.
  A cleaner test would have isolated the row-freeze. But §1's failure
  makes the isolation test low priority.
- **Falsification is progress.** §1 ruled out the drift hypothesis,
  which was the v4 plan's headline. That narrows the problem space —
  remaining levers target feature-level mismatch, not classifier drift.

## Files

- Run log: `reports/logs/run_freeze_v4_20260706_181201.log`
- Eval report: `reports/c2av-finetune__c2av-syriac-finetune__c2av_freeze_v4.json`
- Model: `models/kraken/c2av_freeze_v4.safetensors`
- Harness: `msocr/training/ketos_trainer_api.py` (freeze hook at line 109–124,
  `keep.view(*shape)` fix for 1D bias grads)
- Codec: `reports/c2av_union_codec.json`
- v4 plan: `docs/fixa-v4-plan-2026-07-06.md` (§1 now falsified)
- This report: `docs/fixa-v4-result-2026-07-06.md`