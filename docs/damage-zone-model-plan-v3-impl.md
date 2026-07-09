# Damage-Zone Model — Implementation Plan (v3-impl)

> Implements `docs/damage-zone-model-plan-v3.md` (research plan) and supersedes
> the implementation sections of `docs/fragmented-MS-model-training-plan.md`.
>
> **Validation status:** v3's research is sound. Two tooling details were
> verified wrong *as written* but correct *in spirit* — the CLI mask flag and
> the `orli_train` framing. Both are corrected below. The four-phase approach
> (mask gate, lacuna-aware recognizer, Kraken seg fine-tune, YOLO detector)
> is workable as specified; this doc turns it into runnable, ordered work.

**Status:** implementation plan, not yet executed.
**Date:** 2026-07-09
**Owner:** msocr

---

## 1. Validation of v3 against the installed codebase

### 1.1 Verified correct (proceed as written)

| v3 claim | Verification (file:line / command) |
|---|---|
| Kraken 7.0.2 current; `--valid-regions`/`--merge-regions` removed | `ketos segtrain --help` — no such flags; `ketos --config FILENAME` exists |
| `region_class_mapping` + `line_class_mapping` are the YAML keys | `kraken/ketos/segmentation.py:249-253` reads both from `ctx.params`; `kraken/ketos/ro.py:154-156` |
| YAML keys live at the subcommand param level (not under a `segtrain:` block) | `segmentation.py:233` `params = ctx.meta.copy()` then `params.update(ctx.params)` — flat keys |
| `segtest` reports pixel IoU per class for regions | `kraken/ketos/segmentation.py:523` `Table(... 'Pixel Accuracy', 'IOU', 'Object Count')`; `class_iu` at 524 |
| `segtest` ALSO reports baseline-detection F1 (so v3 §4 worry is overstated) | `segmentation.py:533-545` `bl_f1`, `bl_precision`, `bl_recall`, `bl_detection_per_class` |
| Annotation state stores region polygons + Segmonto types | `msocr/data/session_manager.py:485` `{"id","polygon","type"}`; export at `:1287-1292` |
| PAGE-XML export emits region `<Coords points>` | `session_manager.py:1291-1292` |
| Per-codepoint confidences already extracted in two places | `msocr/training/dump_preds.py:99`; `msocr/service/annotation_api.py:622` |
| Aggregate helpers (`min`/`p10`/`mean`) already exist | `msocr/training/bootstrap_ocr.py:12-21` |
| Line-level image degradation already ported (Kanungo/ocropy) | `msocr/training/line_augment.py` (`ocropy_degrade`, `degrade_line`, `distort_line`) |
| `ultralytics` is NOT a dependency (Phase 2.5 must add it) | `pyproject.toml` grep = 0 matches |
| No synthetic-lacuna / blanking augmentation exists yet | `line_augment.py` only does noise+distortion, no hole/tear blanking |

### 1.2 Verified wrong as written — corrected here

**(A) Phase 0 CLI flag does not exist.** v3 §4 Phase 0 says
`kraken segment -bl -m mask.png`. The real `kraken segment` CLI has
`-m, --maxcolseps INTEGER` (column separators), **not** a mask flag.
There is no CLI path for inference-time mask gating in Kraken 7.0.

The mask gate **does** exist at the Python API level:
`kraken.blla.segment(im, mask=...)` (`kraken/blla.py:60,116`) zeroes
`tensor_im` where `mask==0` before segmentation. The high-level
`SegmentationTaskModel.predict(im, config)` (`kraken/tasks/segmentation.py:81`)
that `msocr/models/inference.py` currently uses does **not** expose `mask`.

**Correction:** Phase 0 must drop to the low-level `kraken.blla.segment`
path, not the TaskModel wrapper. This is a real change to `inference.py`,
not a one-liner CLI flag. See §2.1.

**(B) `ketos orli_train` DOES exist.** v3 §2.4 says it "does not exist."
Verified: `ketos orli_train` is present in 7.0.2
(`ketos --help` lists it; `ketos orli_train --help` runs). However, v3's
**conclusion** is still correct: it trains "an object detection model from
XML facsimile files" keyed on **baseline arc-length points** (the
`--baseline-num-points` flag), not typed regions. It solves line detection
+ ordering, not DamageZone region typing. So: the subcommand exists, v3's
premise is wrong, v3's *recommendation* (don't build on Orli for region
typing) stands. **No plan change needed** — just record the correction.

### 1.3 New facts v3 didn't surface

- The `KetosTrainer` class (`msocr/training/ketos_trainer.py:31`) already
  accepts a `config: dict` matching the ketos YAML schema, but the
  orchestrator (`orchestrator.py:316-346`) **ignores it** and builds CLI
  args directly. The YAML-config path is half-wired; finishing it is the
  Phase 2 integration point.
- No `segtrain` / `segtest` invocation exists anywhere in `msocr/*.py`
  (grep = 0). Phase 2 wires it from scratch.
- `msocr/models/inference.py:158` reads the scalar `line.confidence`, not
  the per-char `rec.confidences` list. Phase 0.5 must reach into the
  recognition record, not the line object, for Borkar-Smith log-prob
  flagging.
- No held-out damage eval manifest exists in `data/manifests/`. Phase 1
  builds the first one.
- No DamageZone annotations exist in code or tests — `"DamageZone"` appears
  only in `docs/` and the frontend vocabulary. The backend round-trips any
  type string the frontend sends, so annotations may exist in live session
  state but have not been mined yet.

### 1.4 Verdict

**v3 is workable.** The four approaches stand. Two corrections needed:
1. Phase 0 uses the `kraken.blla.segment(im, mask=...)` Python API, not
   the `kraken segment -m` CLI flag (v3 was wrong about the CLI).
2. Phase 0.5 reads `rec.confidences` (per-char) from the recognition
   record, not the scalar `line.confidence`.

Everything else in v3 (the experiment matrix, the eval protocol, the
YAML config, the parallel-phase ordering) is correct as written.

---

## 2. Implementation plan — ordered phases

Dependency graph:
```
Phase 1 (data export) ──┬──> Phase 2 (Kraken segtrain)
                        ├──> Phase 2.5 (YOLO seg)
                        └──> Phase 3 (SAM2 adapter, conditional)
Phase 0.5 (lacuna recognizer)  ── parallel, no dep on Phase 1
Phase 0   (mask gate)         ── depends on Phase 2 OR 2.5 winning model
```

Phase 0.5 and Phase 1 can run in parallel from day one. Phase 0 ships
only after Phase 2 or 2.5 produces a usable mask model — it is the
*consumer* of whichever detector wins, not a parallel track.

### Phase 1 — Data export & held-out eval set

**Goal:** turn live `session_manager.py` annotation state into the three
training/eval formats every downstream phase needs, plus the first
damage-specific held-out manifest.

**Why first:** every experiment in §4.5 of v3 consumes the same Phase 1
export. No annotation export → no experiments.

**Scope:**

1. **Export module** `msocr/data/damage_export.py` (new, ~150 LOC):
   - `export_region_polygons(session_ids) -> records` — pulls
     `annotations_v2["regions"]` from `session_manager`, filters to
     regions where `type == "DamageZone"` (and any tracked subclasses),
     returns `{image_path, polygons: [[(x,y),...]], folio_id}`.
   - `write_page_xml(records, out_dir)` — already exists via
     `session_manager._export_page_xml_v2`; call it, don't reimplement.
   - `write_yolo_seg(records, out_dir)` — reformat polygons to YOLO-seg
     `labels/*.txt` + `images/*.png`; a coordinate reformat, not a
     relabel. One file per folio.
   - `write_seg_lst(records, out_path)` — `.lst` files listing
     `image.png\txml_path` pairs for `ketos segtrain --training-data`.
   - `rasterize_damage_mask(record, size) -> PIL.Image('1')` —
     `ImageDraw.Draw(mask).polygon(points, fill=1)`; **this is the one
     new rasterization primitive** v3 needs. Used by Phase 0 gate and
     Phase 2.5 box→mask conversion.

2. **Held-out damage manifest** `data/manifests/damage-holdout.json`
   (new):
   - Deliberately oversample damaged folios into the holdout (v3 §5,
     per YALTAi's own caution — don't random-split and hope).
   - Stratify by severity if the annotation UI captures it; otherwise
     single `DamageZone` class (v3 §4 v1 default).
   - Target: ≥10 damaged folios held out, ≥30 in train. Below 30 train,
     Phase 2/2.5 results will be noisy — log the count, don't block.
   - Manifest schema reuses `manifest.py`'s `partitions` shape
     (`train`/`validation`/`holdout` keyed by `manuscript_id`).

3. **CLI** `msocr export-damage --out-dir data/damage/ --split manifest.json`:
   - Wires export module to the manifest; one command produces all three
     formats in parallel subdirectories.

**Acceptance:**
- `export-damage` runs end-to-end on current session state without error.
- All three format directories populate; folio count logged.
- A sanity `pytest` asserts: train+val+holdout folio counts match the
  manifest; every holdout folio has ≥1 DamageZone polygon.
- Held-out set reviewed by hand for damage representation (not automated).

**Deliverables:** `msocr/data/damage_export.py`, `msocr/cli.py`
`export-damage` subcommand, `data/manifests/damage-holdout.json`,
`tests/data/test_damage_export.py` (3-4 tests, no framework beyond
pytest).

**Effort:** 1–2 days. This is the critical-path bottleneck; everything
else waits on it.

---

### Phase 0.5 — Lacuna-aware recognizer (parallel, no Phase 1 dep)

**Goal:** per-line damage flag from the recognizer's own confidence,
free to build alongside Phase 1 since it reuses the existing HTR
training data (the Sophro Mhiro fine-tune line set).

**Scope:**

1. **Synthetic-lacuna augment** `msocr/training/lacuna_augment.py` (new,
   ~80 LOC):
   - `blank_span(line_img, transcript, frac_range=(0.1,0.4), p=0.3)`:
     pick a contiguous span of width `frac * line_width`, fill with
     background color (mean of corner pixels, not pure white — real
     holes show parchment, not void), and insert a Leiden-style `[...]`
     token at the corresponding position in the transcript.
   - Geometry: prefer horizontal spans (holes/tears) over random-pixel
     blanking — v3 §4 risk note: synthetic gaps won't match real damage,
     but horizontal spans are a better first-order approximation than
     salt noise.
   - Wire into `ketos_trainer.py`'s augmentation pipeline alongside
     `degrade_line` from `line_augment.py`.

2. **Fine-tune** the existing recognizer with augmentation on via
   `KetosTrainer.train_model()` (already wired). No new CLI — this is
   ordinary `ketos train` with `--augment` + the new lacuna pass.

3. **Confidence flag at inference** `msocr/models/lacuna_flag.py` (new,
   ~40 LOC):
   - `flag_line(rec) -> (bool, float)` — reads `rec.confidences`
     (per-char list, already extracted in `dump_preds.py:99`),
     aggregates via `bootstrap_ocr._aggregate(method="p10")` (already
     exists), flags lines below a calibrated threshold.
   - Threshold calibrated against the Phase 1 held-out set: sweep
     threshold, pick the one matching Borkar-Smith's 53% lacuna /
     84% other-error operating point as a floor.

4. **CLI** `msocr lacuna-flag --model X --image Y --threshold Z`:
   - Prints flagged line spans; optional `--json` for pipeline use.

**Acceptance:**
- Flagged-line precision/recall on the held-out damage set, reported as
  a single number pair. Beat the 53%/84% Borkar-Smith floor as a sanity
  check (Sogdian's narrower domain should exceed a generic historical
  baseline).
- One `pytest` asserting `flag_line` returns `True` on a synthetically
  blanked line and `False` on a clean line.

**Deliverables:** `msocr/training/lacuna_augment.py`,
`msocr/models/lacuna_flag.py`, CLI subcommand, one test file.

**Effort:** 1–2 days. Can start immediately, in parallel with Phase 1.

---

### Phase 2 — Kraken `segtrain` fine-tune (after Phase 1)

**Goal:** the cheap baseline. Fine-tune `blla.mlmodel` with a
`DamageZone` region class. Expected to underperform on rare/irregular
damage (per YALTAi's 0.0 mAP finding) but the negative result is
informative and nearly free.

**Scope:**

1. **YAML config** `configs/damage_segtrain.yaml` (new):
   - v3 §4's structure is correct; confirmed keys
     `region_class_mapping` (list of `[name, idx]` pairs, parsed at
     `segmentation.py:252-253`).
   - `load: models/kraken/blla.mlmodel` (or whatever the local blla
     path is — verify before running).
   - `training_data`/`evaluation_data` point at the `.lst` files Phase
     1 produced.
   - `format_type: xml`, `weights_format: safetensors`.

2. **Wire `segtrain` into the trainer** `msocr/training/ketos_trainer.py`
   (extend):
   - Add `seg_train_model(self, config_yaml)` mirroring `train_model()`
     but invoking `ketos --config <yaml> segtrain` via subprocess (same
     pattern as the existing `train` call at line 139).
   - Add `seg_test_model(self, ...)` invoking `ketos segtest` and parsing
     the pixel-IoU table (the metric v3 §5 wants, confirmed at
     `segmentation.py:523`).

3. **Orchestrator hook** `msocr/training/orchestrator.py` (extend):
   - New `walk_damage_seg()` function (sibling to `walk_style_group()`)
     that: enriches XMLs with polygon Coords (reuse
     `_enrich_xml_with_polygons`) → runs `seg_train_model` locally or
     via `RunPodRunner` → downloads `best_*.safetensors` → runs
     `seg_test_model` → logs IoU.

4. **RunPod path** `msocr/training/runpod_runner.py` (extend):
   - The existing `run_training()` uploads XML+images and runs a `ketos`
     command on a GPU pod. Add a `segtrain` mode that swaps the command
     string and uploads the YAML config alongside the data. No new pod
     template — same image.

**Acceptance:**
- Training run completes on the held-out set without error.
- `segtest` reports per-class pixel IoU for `DamageZone`; log the
  number. (v3 §4 worried segtest might not report IoU — verified it
  does, `segmentation.py:523,524`.)
- If DamageZone IoU ≥ 0.4, this is a candidate for the Phase 0 mask
  source. If < 0.1 (expected), document the negative result and move to
  Phase 2.5.

**Deliverables:** `configs/damage_segtrain.yaml`,
`ketos_trainer.py:seg_train_model/seg_test_model`,
`orchestrator.py:walk_damage_seg`, `runpod_runner.py` segtrain mode,
one smoke test that runs segtrain on 5 synthetic folios.

**Effort:** 2–3 days. Most of it is wiring; the YAML is 15 lines.

---

### Phase 2.5 — YOLOv8-seg object detector (after Phase 1, parallel with Phase 2)

**Goal:** the best-evidenced bet for rare/irregular classes (YALTAi's
45–100 mAP on rare classes vs Kraken's 0.0). Runs in parallel with
Phase 2 on the same held-out set.

**Scope:**

1. **Add dependency** `ultralytics` to `pyproject.toml`. Pin a recent
   stable version; `yolov8n-seg` and `yolov8s-seg` are the targets.

2. **YOLO trainer** `msocr/training/yolo_trainer.py` (new, ~120 LOC):
   - `train(data_yaml, model="yolov8n-seg.pt", epochs=150, imgsz=1280,
     patience=30)` — wraps `ultralytics.YOLO(...).train()`. Start from
     `n` (nano) per v3 §4.5 — small data, small model, faster converge.
   - `predict_and_mask(model_path, image) -> PIL.Image('1')` — runs
     inference, rasterizes detected polygons to a binary mask (reuse
     `damage_export.rasterize_damage_mask`).

3. **Integration with orchestrator** — new branch in `walk_damage_seg`
   or a sibling `walk_damage_yolo()` that runs the YOLO trainer on the
   same Phase 1 export, in parallel with the Kraken segtrain run.

4. **Inference path** `msocr/models/inference.py` (extend):
   - Add a `damage_mask` pre-pass: if a YOLO damage model is configured
     (env `MSOCR_DAMAGE_MODEL_PATH`), run it to produce a mask, then
     feed that mask into the Kraken segmentation via the
     `kraken.blla.segment(im, mask=...)` low-level API (Phase 0's
     correction, §1.2(A) above).
   - This is where Phase 0 actually ships — YOLO produces the mask, the
     mask gates Kraken's blla segmenter.

**Acceptance:**
- mAP@0.5 on the held-out damage set (matches YALTAi's reported metric
  for direct comparison).
- IoU on the held-out set (matches PLM-SegFormer's 51.2% benchmark,
  computed independently — Ultralytics reports it natively).
- Whichever of Phase 2 / Phase 2.5 scores higher on the same held-out
  set becomes the Phase 0 mask source. The other is documented with
  its actual numbers as a rejected alternative.

**Deliverables:** `msocr/training/yolo_trainer.py`,
`msocr/models/inference.py` mask-gate path, `pyproject.toml`
`ultralytics` dep, `tests/training/test_yolo_trainer.py` (smoke test
on synthetic data).

**Effort:** 3–4 days. New dependency + new inference path is the bulk.

---

### Phase 0 — Mask-gated inference (after Phase 2 OR 2.5 wins)

**Goal:** feed the winning damage model's mask into Kraken segmentation
so damaged bands are skipped during line detection.

**Why last:** v3 framed Phase 0 as "use existing polygons, minimal
effort." That framing was wrong about the CLI (no `-m mask.png` flag).
The real Phase 0 is: take whichever detector won in Phase 2/2.5, run
it per-folio at inference time, rasterize its output to a mask, pass
that mask to `kraken.blla.segment(im, mask=mask)`. It is the
*integration* step, not a standalone phase.

**Scope:**

1. **Mask-gated segmenter** `msocr/models/inference.py` (extend — same
   change as Phase 2.5 step 4; they're one unit of work):
   - Replace the current `SegmentationTaskModel.predict(im, config)`
     call (`inference.py:115-118`) with a path that:
     1. Runs the damage detector (YOLO or Kraken seg, whichever won) →
        produces `damage_mask: PIL.Image('1')`.
     2. Calls `kraken.blla.segment(im, mask=damage_mask, ...)` directly,
        bypassing the TaskModel wrapper (which doesn't expose `mask`).
     3. Feeds the resulting segmentation into the existing recognition
        path (`inference.py:122-126`).
   - Gate this behind an env flag (`MSOCR_DAMAGE_GATE=1`) so default
     behavior is unchanged when no damage model is configured.

2. **End-to-end CLI** `msocr htr --lang sogdian --model X --damage-model Y
   --image Z.png`:
   - Existing `htr` subcommand gains `--damage-model` (optional). When
     set, the mask-gated path runs; when absent, current behavior.

**Acceptance:**
- On a held-out damaged folio: mask-gated run produces fewer
  hallucinated-across-lacuna transcriptions than ungated run. Measure
  as CER delta on the subset of lines that cross a DamageZone polygon
  (vs lines that don't — the latter should be unchanged).
- One `pytest` that mocks the damage model and asserts the mask is
  passed to `kraken.blla.segment`.

**Deliverables:** `inference.py` mask-gate path, `cli.py --damage-model`
flag, one integration test.

**Effort:** 1–2 days. The hard part (mask rasterization, detector
inference) is done by Phase 2.5 step 4; Phase 0 is the final wiring +
the acceptance CER measurement.

---

### Phase 3 — SAM2 adapter (conditional, only if 2 and 2.5 both plateau)

**Goal:** MapSAM2-style adapter on a frozen SAM2 encoder, the
lowest-annotation-floor option. Only justified if Phase 2 AND 2.5 both
plateau below acceptable accuracy AND Phase 1's annotated-folio count
is well below 30.

**Scope:** Deferred until the Phase 2/2.5 eval numbers are in. v3 §4.5
Experiment C has the spec; no implementation work now. If triggered:
custom adapter layers + training loop, ~1–2 weeks, closest in shape to
prior LeWM/GAIL work. Not planned in detail here — revisit with real
numbers in hand.

---

## 3. Shared eval protocol (applies to Phase 2 and 2.5)

Per v3 §4.5, both experiments report on the same held-out set:

- **mAP@0.5** — matches YALTAi's reported metric; Ultralytics emits
  this natively. For Kraken segtrain, compute independently from the
  segtest confusion output (segtest gives pixel IoU, not mAP, so
  derive mAP from per-folio precision/recall at IoU≥0.5).
- **IoU** — matches PLM-SegFormer's 51.2% mIoU. Kraken `segtest`
  reports it directly (`segmentation.py:523`); Ultralytics reports
  seg-IoU natively.
- **Wall-clock + GPU-hours** — log alongside accuracy; cost-per-mAP
  is the tiebreaker if the two land close.

Both metrics go into a single `docs/damage-eval-results.md` after the
runs land, so the Phase 0 decision (which model gates the mask) is
made on the same page, not from a tool's default report.

---

## 4. Decision points to resolve before starting

1. **DamageZone subclass granularity.** v1 default is single collapsed
   `DamageZone`. Confirm before Phase 1 export — if the frontend already
   distinguishes hole/tear/stain/foxing, capturing it now is free;
   collapsing later is lossy.
2. **Held-out folio count.** How many damaged folios are actually in
   session state today? Phase 1's first step is a count query against
   `session_manager`; if it's <10, the whole plan slows and Phase 3's
   low-annotation-floor argument strengthens.
3. **RunPod budget for parallel 2 + 2.5.** v3 §7.6 flags this. Both
   consume GPU; confirm the A100/H100 budget covers simultaneous runs
   or sequence them.
4. **blla.mlmodel local path.** Phase 2's YAML `load:` needs the actual
   path on disk; verify it exists under `models/kraken/` or wherever
   the default blla weights landed after `uv sync`.

---

## 5. Risk deltas beyond v3 §6

- **Phase 0 CLI assumption (resolved).** v3's `kraken segment -m mask.png`
  is wrong; corrected to the `kraken.blla.segment(im, mask=...)` Python
  API. No plan risk remaining — just the implementation cost of
  swapping the TaskModel wrapper for the low-level call.
- **`orli_train` framing (resolved).** v3 §2.4 says it doesn't exist; it
  does, but it doesn't do typed regions, so v3's recommendation stands.
  No plan impact.
- **Frontend DamageZone usage.** No backend code references
  `"DamageZone"` literally. If the frontend isn't actually emitting it
  in practice, Phase 1 will find zero damaged folios and the plan
  stalls. **Mitigation:** Phase 1 step 0 is a `session_manager` audit
  query counting regions by `type`; run it before anything else.

---

## 6. Summary table

| Phase | Dep | New files | Effort | Output |
|---|---|---|---|---|
| 1 Data export | — | `msocr/data/damage_export.py`, CLI `export-damage`, manifest | 1–2d | PAGE-XML + YOLO-seg + .lst + held-out manifest |
| 0.5 Lacuna recognizer | — | `msocr/training/lacuna_augment.py`, `msocr/models/lacuna_flag.py`, CLI | 1–2d | Per-line damage flag, free sanity signal |
| 2 Kraken segtrain | 1 | `configs/damage_segtrain.yaml`, trainer + orchestrator hooks | 2–3d | Cheap baseline IoU/mAP (expected low) |
| 2.5 YOLO seg | 1 | `msocr/training/yolo_trainer.py`, `ultralytics` dep, inference mask path | 3–4d | Expected winning detector |
| 0 Mask gate | 2 or 2.5 | `inference.py` blla-mask swap, `--damage-model` flag | 1–2d | End-to-end mask-gated HTR |
| 3 SAM2 | 2+2.5 plateau | custom | 1–2w | Only if both plateau |

**Critical path:** Phase 1 → (Phase 2 ∥ Phase 2.5) → Phase 0.
Phase 0.5 runs free in parallel from day one.

Total: ~8–11 dev-days to a working mask-gated pipeline, assuming
Phase 1 finds ≥30 damaged folios in session state.