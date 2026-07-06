# Fix A v4 — Implementation Plan (2026-07-06)

Companion to `docs/fixa-v4-plan-2026-07-06.md`. The plan doc diagnosed the
problem (severe overfitting: 6% train CER vs 60–71% val/holdout, 87.3% of
holdout errors on shared classes) and proposed four levers. This doc is the
corrected, executable version: it fixes three technical errors in the plan,
maps each lever to concrete file paths in this repo, and sequences the work
into phases that can be dispatched and verified independently.

## 0. Research corrections to the v4 plan doc

Verified against current Kraken source (`mittagessen/kraken` @ 9a218ce8)
and the live codebase (`msocr/`).

| § | Plan claim | Finding | Action |
|---|---|---|---|
| 1 | `--freeze-backbone` freezes everything except the classifier | ✅ Confirmed. `KrakenFreezeBackbone.on_train_start` calls `self.freeze(pl_module.net[:-1])`, leaving `LinSoftmax` trainable. CLI help: "Number of samples to keep the backbone (everything but last layer) frozen." | Plan stands. |
| 1 | Per-row gradient hook on `weight`/`bias` freezes old rows | ✅ Sound **for this recipe**. Attribute path: `model.nn[-1].lin.weight` shape `[num_classes, hidden]`, `.bias` shape `[num_classes]`. `tensor.register_hook(lambda g: mask*g)` is the right mechanism. Softmax-renormalization shifts old-class probabilities as new rows learn — that is the *desired* signal, not drift; with backbone fully frozen (`999999`), old-row weights are literally constant once their grads are zeroed. | Keep. |
| 2 | `ketos linegen -m -s -r -d` flags | ❌ Old-version flags. Current Kraken uses `--alpha --beta --distort --distortion-sigma`; `-d` is `--disable-degradation` (bool). | Fix invocation (see §4 below). |
| 2 | U+074D/074E/074F are "recent, narrow"; Noto may lack them | ⚠️ Overstated. They are Unicode 4.0 (2003), and **Noto Sans Syriac covers them** (already installed at `/usr/share/fonts/truetype/noto/NotoSansSyriac-Regular.ttf`). | Font check becomes a 1-minute programmatic verification, not a blocker. Paleographic plausibility is still a human call. |
| 3 | "Crank up `--augment`" | ❌ `--augment` is boolean-only; fixed `DefaultAugmenter` albumentations pipeline (pixel dropout, blur, optical distortion, elastic, ±3° rotate), applied with p=0.5. No intensity knob, no swappable augmenter. | §3 must be **offline pre-degradation** of the 148 real lines, not a CLI flag. |

## 1. Codebase gaps the research exposed

These do not exist in the repo today and must be built:

- **No codec-dump script.** The plan says "the codec dump from the 0.1/3.1
  audit work already has this mapping" — it doesn't. `scripts/confusion_analyze.py:27-28`
  hardcodes the 5 new-class chars but doesn't dump the row-index map. Step 0a
  builds it.
- **No Python-API training path.** `msocr/training/ketos_trainer.py:53`
  shells out to `ketos` CLI via `subprocess.run`. The §1 freeze-hook fix
  needs a Python-API harness driving `KrakenTrainer` (Lightning) directly so
  the hook fires — but it must still emit `best_*.safetensors` to the path
  the CLI would, so `scripts/triangulate_eval.py`,
  `scripts/confusion_audit.py`, and `msocr/evaluation/harness.py` work
  unchanged.
- **No `ocropy_degrade`/`distort_line` usage anywhere.** §3 needs an
  offline pre-degradation script.
- **No automated annotation→training manifest path.** Annotation API
  (`msocr/service/annotation_api.py`) exports PAGE XML to the session dir,
  but moving it to `dataset/christian_sogdian_c2av/gt/` and editing
  `data/manifests/c2av-finetune.json` is manual. This is the friction behind
  "including my annotation jobs" — Phase A removes it.

## 2. Current state (verified by exp-1)

- **Manifest**: `data/manifests/c2av-finetune.json` — train: c2av01–c2av10
  (10 plates, 148 lines), val: c2av11 (15 lines), holdout: c2av12 (19 lines).
- **GT**: `dataset/christian_sogdian_c2av/gt/c2av{01..12}_page_{01..12}.xml`
  (Syriac script, converted from Latin by `scripts/latin_to_syriac.py`).
- **Plate images**: `dataset/christian_sogdian_c2av/plates/p-*.png` (99
  total; 12 annotated, 87 remaining).
- **v3 shipped recipe** (`docs/fix-a-v3-syriac-script-finetune.md:77-78`):
  `--resize union --freeze-backbone 999999 --augment --warmup 200 --lr 1e-4
  --epochs 200 --lag 25 --min-epochs 8`. Holdout CER 71.43%.
- **5 new classes** (after `--resize union`, 37 old + 5 new = 42-way head):
  U+0741 qushshaya (192 occ), U+0742 rukkakha (11), U+074D zhain (10),
  U+074E khaph (57), U+074F fe (10). Hardcoded in
  `scripts/confusion_analyze.py:27-28` as the cross-check source.
- **Lexicon**: `dataset/lexicon/sogdian_words.txt` (7,387 unique words,
  extracted from `NSW2021corpus.xls`).
- **Annotation API**: `msocr/service/annotation_api.py` on port 8001,
  serves React SPA from `frontend/dist/`, exports PAGE/ALTO/TSV.

## 3. Phased implementation

### Phase 0 — Scaffolding (no training, ~30 min)

**0a. `scripts/dump_codec.py`**
- Load base model via `kraken.lib.vgsl.TorchVGSLModel.load_model(path)`.
- Iterate `model.codec.c2l` (char→row map), print human-readable table.
- Flag the 5 new-class rows (intersection with the hardcoded set in
  `scripts/confusion_analyze.py:27-28`).
- Emit `reports/c2av_union_codec.json`:
  `{"old_rows": [...], "new_rows": [...], "char_to_row": {...}}`.
- **Verify assumption**: Kraken's `PytorchCodec` sorts by codepoint, so the
  5 new classes (highest codepoints in the union) should be the last 5
  rows. The script confirms rather than assumes.

**0b. `msocr/training/ketos_trainer_api.py`**
- New module. Exposes `train_with_row_freeze(config, xml_files, freeze_old_rows: bool)`.
- Uses `KrakenTrainer` (Lightning) directly — not `subprocess.run` — so the
  backward hook fires. Mirrors `KetosTrainer.train()`'s compile/resize/train
  sequence but at the Python API level.
- Load model → `--resize union` (call `model.resize_output(42, 'union')` or
  equivalent) → locate `model.nn[-1].lin.weight` / `.bias` → read
  `reports/c2av_union_codec.json` for old-row indices → register
  `tensor.register_hook(lambda g: mask_rows(g, old_indices))` on both
  tensors. The hook zeros `.grad` rows for old classes before the optimizer
  step.
- Hook persistence: register once at load time; if PyTorch drops the hook
  after first backward (it doesn't, but verify), re-register in a
  `on_train_epoch_start` callback.
- Emits `best_*.safetensors` to the same output dir the CLI would, so
  downstream eval (`triangulate_eval.py`, `confusion_audit.py`,
  `harness.py`) is unchanged.
- Reuses the existing `RunPodRunner` patches (`_KRAKEN_CHECKPOINT_PATCH`,
  `_SAFETENSORS_SHARED_TENSOR_PATCH`) — those fix Kraken bugs that also
  bite the Python-API path.

**0c. CLI wiring**
- `msocr train --freeze-old-rows` flag (boolean), passed through to
  `KetosTrainer` / the new API harness.
- `msocr train-remote --freeze-old-rows` flag, mirrored through
  `orchestrator.py:223-253` into the RunPod `ketos train` invocation (which
  for the API path means shipping a small Python entrypoint script to the
  pod instead of the raw `ketos` CLI).
- Existing CLI paths (no `--freeze-old-rows`) stay default so v3's run
  remains reproducible bit-for-bit.

**0d. Font-coverage check (for Phase 4 prep)**
- `scripts/check_font_coverage.py` — uses `fontTools.ttLib.TTFont` +
  `getBestCmap()` to verify `/usr/share/fonts/truetype/noto/NotoSansSyriac-Regular.ttf`
  covers U+0741, U+0742, U+074D, U+074E, U+074F.
- `scripts/render_font_sample.py` — renders one test line per rare
  codepoint (zhain, khaph, fe, qushshaya, rukkakha) via `ketos linegen` to
  `reports/font_samples/` for human visual QA. **This is the user's
  paleographic call** — glyph presence ≠ ductus match for the c2av hand.

### Phase 1 — §1 classifier-row freeze run (1 GPU-hour on RunPod)

- Identical v3 recipe, only change: `--freeze-old-rows` on top of
  `--resize union --freeze-backbone 999999 --epochs 200 --lag 25 --lr 1e-4
  --augment --warmup 200 --min-epochs 8`.
- Dispatched via `msocr train-remote` to RunPod (RTX 3090, ~1 hour).
- Downloaded `best_*.safetensors` → `models/kraken/c2av_finetune_v4_freeze.safetensors`.
- **1b. Re-triangulate**: `scripts/triangulate_eval.py` on c2av01/11/12.
- **1c. Re-audit confusion**: `scripts/confusion_audit.py` +
  `confusion_analyze.py` on c2av12 holdout. Compare shared-class-error % to
  the 87.3% baseline.
- **Decision gate**:
  - If shared-class errors collapse AND train/val gap narrows → hypothesis
    confirmed, ship the freeze as the new default.
  - If not → drift hypothesis falsified; the manuscript-hand mismatch
    (v3's §3.3 scenario) is the real cause, and §2 synthetic + more
    annotation (Phase 4 + Phase A) become the only levers.

### Phase 2 — §3 offline pre-degradation (parallel to Phase 1, no GPU)

**`scripts/augment_real_lines.py`**
- For each of the 148 real line crops (extracted from
  `dataset/christian_sogdian_c2av/gt/c2av{01..10}_page_*.xml` + plate
  images), apply `kraken.linegen.ocropy_degrade` (combined) and a
  `distort_line` + `degrade_line` variant, N times (start N=3), with
  randomized params (`distort`, `dsigma`, `eps`, `delta` jittered ±20%).
- Write augmented crops + duplicated PAGE-XML `<TextLine>` entries into
  `dataset/christian_sogdian_c2av/gt_aug/` (sibling tree, original GT
  untouched).
- Emit `data/manifests/c2av-finetune-aug.json` — train partition references
  real + augmented lines (148 × 4 = 592 train lines); val/holdout
  unchanged (c2av11/c2av12 real lines only).
- Train Phase-1 recipe on the augmented manifest; triangulate. If train/val
  gap narrows without holdout regressing → overfitting-by-memorization
  confirmed; stack this on §1 as cheap insurance.

### Phase 3 — §4 leave-one-plate-out CV + ensemble (after §1+§2 settle)

**`scripts/loo_cv.py`**
- For each of the 12 plates, build a fold manifest (that plate as holdout,
  remaining 11 split train/val by the existing 10/1 logic).
- Submit 12 Phase-2-recipe runs to RunPod (parallelizable across pods).
- Collect 12 holdout CERs; report mean ± spread.
- **Output**: `reports/loo_cv_c2av.json` + `reports/loo_cv_c2av.md`.
- If 71.43% is inside the band → it's representative. If wide → single-plate
  holdout was noise; report the band as the new number.

**`scripts/ensemble_decode.py`**
- Load all 12 fold models; average logits; majority-vote decode on c2av12
  and the full corpus.
- Standard no-new-data way to claw back accuracy from independently-overfit
  models.

### Phase 4 — §2 synthetic data via `ketos linegen` (gated on user)

Corrected invocation (current Kraken flags):
```bash
ketos linegen -f "Noto Sans Syriac" -u NFC \
  --alpha 1.5 --beta 1.5 --distort 3.0 --distortion-sigma 10 \
  -o training_data_sogdian synth_source.txt
```

- Source text: bias toward words containing the 5 rare classes (rukkakha,
  zhain, fe especially — 11/10/10 occurrences) from
  `dataset/lexicon/sogdian_words.txt`. Upweight rare-class words 10×.
- Convert Latin→Syriac via `scripts/latin_to_syriac.py` if needed (lexicon
  may already be in Syriac script — verify at build time).
- Mix synthetic lines into **train only**; val/holdout untouched.
- **Gated on**: user paleographic QA of `reports/font_samples/` from
  Phase 0d. If Noto glyphs look wrong for the c2av hand, try academic
  Syriac fonts (Serto, East Syriac Adiabene, Estrangelo) before generating.

### Phase A — Annotation-jobs integration (user's work, unblocked)

The v3 plan says real annotation is the only lever with no technical
substitute for CER well below the ceiling. The codebase has a friction
point: exported annotations need manual XML copy + manifest JSON edit.
Phase A removes it.

**`scripts/ingest_annotated_plate.py`**
- Input: a session_id from the annotation API (or a plate filename).
- Calls the existing `/api/sessions/{id}/export?format=page` endpoint on
  the running annotation API (port 8001) — or reads the already-persisted
  PAGE XML from the session dir directly.
- Copies the XML to `dataset/christian_sogdian_c2av/gt/c2avNN_page_NN.xml`.
- **Auto-updates `data/manifests/c2av-finetune.json`** to include the new
  plate in the partition specified by `--partition train|val|holdout`.
- After each batch of newly annotated plates, re-run Phase 1's recipe on
  the expanded manifest. The classifier-row freeze means new real data
  can't drift the old rows, so adding plates is pure upside.
- **User decision needed**: default partition for new plates (train vs
  val vs holdout), and which plates are in progress.

## 4. Suggested dispatch order

1. **Phase 0a–0c** (codec dump + freeze harness + CLI) — agent-executable
   now, ~30 min.
2. **Phase 0d** (font samples) — agent-executable now, ~5 min; user QA
   async.
3. **Phase 1** (freeze run on RunPod) — dispatched after Phase 0a–0c land;
   ~1 GPU-hour.
4. **Phase 2** (offline augmentation) — parallel to Phase 1, no GPU.
5. **Phase 1b/1c** (triangulate + audit) — after Phase 1 downloads.
6. **Phase 3** (LOO-CV + ensemble) — after §1+§2 settle on a recipe.
7. **Phase 4** (synthetic) — gated on Phase 0d user QA.
8. **Phase A** (annotation ingest) — anytime; unblocks the only lever
   with no technical substitute.

## 5. What needs the user, not the agent

- **Phase 0d / Phase 4**: paleographic judgment on whether Noto Sans
  Syriac (or an academic alternative) is a plausible visual match for the
  c2av scribal hand. Glyph coverage is verifiable; ductus match is not.
- **Phase 4**: deciding how much synthetic data to mix in relative to the
  148 real lines, after seeing real numbers from a first small batch.
- **Phase A**: any further real annotation of additional c2av (or other
  Turfan) plates — manual transcription, the only lever with no technical
  substitute. And the default partition for newly annotated plates.

Everything else (the classifier-row freeze, the augmentation script, the
CV/ensemble harness, the `ketos linegen` scripting, the annotation ingest
script) is agent-executable without further input.