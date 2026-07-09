# Fix A — v9 Fragment Pipeline Plan

> v8 → v9: companion annotation doc rewritten as eScriptorium-feature-parity
> plan for our SPA (was: "use SPA, fix small gaps"). Pipeline stages unchanged.
> Original v8 header below.
>
> Validated update to v7. Reconciles the v7 plan against current Kraken 7.0.2
> docs/source, 2024–2026 fragmentary-HTR literature, and the actual state of
> the `msocr` codebase + dataset on disk. supersedes
> [`fix-a-v7-fragment-pipeline.md`](./fix-a-v7-fragment-pipeline.md).
>
> Date: 2026-07-07. Author: orchestrator, with @librarian (claims validation)
> and @explorer (codebase map).

---

## Status (updated 2026-07-07)

| Stage | Status | Owner | Commit / Notes |
|---|---|---|---|
| 0.1 v6 blocker | **DEFERRED (watchdog fixed)** | — | RunPod pod `phpyu59p2flw8g` terminated. MS Jer 36 intermediate adaptation deferred; Stage 4 will use raw `sophro_mhiro_syriac.safetensors` base. Re-spin RunPod when ready for Stage 4. `ssh_exec` PTY/headless-block bug **fixed** (uncommitted working tree 2026-07-09): `SSH_EXEC_IDLE_TIMEOUT=120` module constant + idle-timeout watchdog in `runpod_runner.py` — force-closes channel+client when `last_recv_time` silent >`idle_timeout` AND `exit_status_ready()` is False, raises `RuntimeError`. Test `test_ssh_exec_raises_on_idle_timeout` added. The PTY bug that hung `ketos train` on RunPod is now bounded to 120 s instead of hanging forever. |
| 0.2 LOOCV | **BLOCKED on GPU** | — | Needs RunPod + 12 model trains. Deferred until Stage 4. 1/12 folds done (`c2av01`, CER 77.9%). |
| 0.3 Fragment convention backend | **DONE** | fix-2 | Commit `b0fd817`. RTL `<ReadingOrder>` parse+emit, multi-region nesting, ALTO kept+fixed, orphan-line nearest-region fallback. 8 tests pass. |
| 0.4 Annotation scale decision | **DONE** | user | "Annotate more" — target ~50 plates / ~750 lines. Gates Stage 3 (≥30 plates) and Stage 4. |
| 1 Preprocessing wiring | **DONE** | fix-3 | New `msocr/preprocessing/pipeline.py` (116 lines) chains isolate → binarize → deskew → manuscript_area → row_bands. Self-check passes (`uv run python -m msocr.preprocessing.pipeline` → `ok`). No preprocessing/segmentation tests broke. Per-fragment `_mask.png` is pre-deskew (used for Hough); BLLA re-binarizes from deskewed crops at Stage 3. |
| 2.1-2.2 Within-line / DamageZone policy | **DONE** | fix-1 | Tier 0 frontend: U+0323 in palette, DamageZone default-visible. |
| 2.3 dump-preds + /bootstrap | **DONE** | fix-4 | New `msocr/training/dump_preds.py` (261 lines, generalizes `scripts/dump_c2av12_preds.py`), `msocr dump-preds` CLI subcommand (+38 lines in `cli.py`), POST `/api/sessions/{id}/bootstrap` endpoint (+137 lines in `annotation_api.py`). Builds kraken `Segmentation` from v2 baselines+boundaries via `BaselineLine`, loads model once, predicts per line, means per-char confidences into scalar. Self-check ok; 35 CLI/service/data tests pass. Frontend wiring (calling `/bootstrap` from `AnnotateEditor`) is Tier 2 #17-20 work. |
| 3 BLLA seg fine-tune | **BLOCKED on annotation** | — | Needs ≥30 fragment-aware plates. ~50-plate batch unblocks this. |
| 4 Recognition fine-tune | **BLOCKED on GPU + LOOCV** | — | Needs RunPod + Stage 0.2 LOOCV first. |
| 5 Inference + post-correction | **DEFERRED** | — | After Stage 4. |
| 6 Evaluation | **PARTIAL** | — | `confusion_analyze.py` exists; needs LOOCV report to be meaningful. |

---

## TL;DR — what changed from v7

| # | v7 assumption | v8 correction | Why |
|---|---|---|---|
| 1 | ByT5-small post-correction on ~150 line pairs | **Drop.** Replace with non-neural weighted-Levenshtein + char-confusion matrix. | Every published ByT5 success uses 500–10,000+ pairs. 150 is below the floor; non-neural baselines beat it. (lib-1, Claim 3) |
| 2 | `--resize union` + `--freeze-backbone` for new char classes | **Switch to `--resize new`** + `--warmup` 1–2 epochs + `--freeze-backbone` covering ≥1 epoch. | Kraken docs explicitly warn against `union` for fine-tuning ("network will rapidly unlearn missing labels"). v4's 71.43%→80.09% regression is a documented failure mode. (lib-1, Claim 5) |
| 3 | LOOCV across 12 plates is available | **LOOCV is 1/12 done.** Complete it before any recognizer claim. | `reports/lopo/` has only fold c2av01. v4 lesson: test the measurement before testing the model. (exp-1, finding 8) |
| 4 | Fragment-aware annotation pipeline exists | **It does not.** PAGE XML export emits no `<ReadingOrder>`; the React SPA has no `row_id`/fragment grouping, no Leiden underdot (U+0323) in the palette, DamageZone is hidden behind an "Advanced" toggle, ALTO export ignores v2 state, and all lines nest under the first region. | Our own `docs/kraken-fragmentary-manuscripts.md` §4 convention is not implemented in either the export (`session_manager.py`) or the frontend (`AnnotateEditor.tsx`, 1065 lines). See [`fix-a-v9-annotation-plan.md`](./fix-a-v9-annotation-plan.md) §8-9 for the sized frontend work list. (exp-1, exp-2) |
| 5 | Segmentation pipeline (isolate → deskew → BLLA) is wired | **Standalone modules, never chained.** `deskew.py` exists but `preprocessor.py` still uses whole-page deskew. | exp-1, finding 2 |
| 6 | 12 plates / 178 lines is the dataset | **99 plates exist, 12 annotated.** 87 plates are unannotated. The annotation bottleneck dominates modeling. | exp-1, finding 1 |
| 7 | Orli is "unreleased" | **Orli is published** (arXiv 2606.04166, code on github.com/mittagessen/orli) but not in Kraken mainline. | lib-1, Claim 7 |
| 8 | v6 Stage 1 (MS Jer 36 adaptation) is in progress | **Blocked** — orchestrator stuck on pip install; pod still billing. Resolve before extending. | exp-1, finding 7 |

**Confirmed v7 claims (no change):** `--suppress-regions` exists and is the right knob (lib-1, Claim 1). BLLA's three weak spots (non-Latin, diacritized, fragmentary) stack (lib-1, Claim 2). No per-pixel ignore channel exists in Kraken seg/rec target tensors (lib-1, Claim 6).

---

## Stage 0 — Prerequisites & unblocking (NEW in v8)

v7 assumed a measured baseline and a working annotation pipeline. Neither is true. v8 makes them gates.

### 0.1 Resolve the v6 Stage 1 blocker

The MS Jer 36 adaptation pod is running and billing but `ketos train` never launched — orchestrator is stuck on a pip install command due to an incomplete `ssh_exec` PTY fix (`fixa-v6-stage1-blocker-handoff.md`).

- **Decision:** kill the pod today unless we will resume Stage 1 within 24h. Spot billing at $0.34/hr is cheap but indefinite billing is a leak.
- **Then:** either fix `ssh_exec` (PTY removal caused silent hangs on quiet commands — re-add a bounded PTY with a timeout, or switch to `paramiko.exec_command` with a hard read timeout) OR mark Stage 1 deferred and proceed to Stage 1' (Sogdian-only fine-tune from Sophro Mhiro without the MS Jer 36 intermediate).
- **Ponytail:** the runpod runner is 300 lines and the bug is in one method. Fix the method, don't rewrite the runner.

### 0.2 Complete LOOCV before any recognizer claim

`scripts/build_lopo_manifests.py` + `msocr/evaluation/harness.py` exist and produced 1 fold (`c2av-finetune-lopo-c2av01`, CER 77.9%). The other 11 folds never ran.

- **Run all 12 folds** with the current shipped model (`c2av_finetune_union_frozen.safetensors`, the v2 frozen-forever regime). This is the only honest measurement of where we are.
- **Aggregate** into a single report: mean ± std CER across folds, per-plate CER, deletion vs substitution breakdown (reuse `scripts/confusion_analyze.py`).
- **Gate:** if any fold's holdout CER is below 50%, lexicon post-correction becomes worth re-trying. If all folds are above 70%, the recognizer is the bottleneck, not post-correction — and no amount of ByT5 would have saved it.
- **Do NOT start Stage 4 (recognition fine-tuning) until this report exists.** v4's lesson: we spent a freeze-rows experiment on top of a measurement we didn't have.

### 0.3 Annotation pipeline: implement the fragment convention

`docs/kraken-fragmentary-manuscripts.md` §4 specifies: one baseline per contiguous inked fragment, `DamageZone` regions over lacunae, RTL `ReadingOrder`, Leiden underdot for uncertain glyphs. Neither the PAGE XML export (`session_manager.py` `_export_page_xml_v2`) nor the React frontend (`AnnotateEditor.tsx`) implements these.

**Backend / export (this stage):**
- **Add RTL `<ReadingOrder>` element** to `_export_page_xml_v2`. Currently absent. Parse it back in `parse_page_xml_to_v2`.
- **Fix multi-region nesting:** v2 export currently nests all lines under the first `<TextRegion>`. Track line→region membership and emit correctly.
- **Fix ALTO export:** `_export_alto` uses v1 state only and silently ignores any annotation made via the drawing UI (v2). Either fix to read v2 or delete ALTO export (ponytail: delete unless a consumer needs it).
- `DamageZone` is already in the region vocabulary and exported as a SegmOnto-typed region — no backend work needed there.

**Frontend (see `fix-a-v9-annotation-plan.md` §8-9 for the sized list):** U+0323 in palette (XS), DamageZone visible by default (XS), `row_id` for fragment grouping (M), recognition in `/autosuggest` (M), per-line confidence display (S).

- **Ponytail:** the convention is documented but unimplemented, and every downstream stage depends on it. This is the one place new code is justified. Split the work: backend/export here, frontend in the annotation plan doc.

### 0.4 Annotation scale decision

87/99 c2av plates are unannotated. 178 lines is below every documented minimum (`htr-finetuning-data-practices-2026-07-06.md`): 5-line val is noise, 19-line holdout gives CER CI ±4–7 pts, k-fold CV needs ~142 train / 36 val per fold.

- **Decision point:** annotate more plates, or accept that 178 lines is the floor and optimize within it. This is a human-time question (annotation is slow), not a compute question.
- **If annotate:** target ~50 plates / ~750 lines. At ~15 lines/plate that's ~3× the current data. This is the single highest-leverage action in the whole plan.
- **If accept 178:** every downstream claim must be hedged with wide CIs. LOOCV (Stage 0.2) is the only honest measurement. Do not add a model stage on top of an under-measured 178-line base.

**Cautions from annotation-methodology research (full detail in `fix-a-v9-annotation-plan.md`):**
- **Couture et al. 2023:** at small scale, *protocol consistency* beats *quantity*. Inconsistent graphemic judgment calls between plates inject silent noise. Write the convention down once, apply by checklist. Annotating more plates with inconsistent conventions is worse than annotating fewer plates consistently.
- **Mohammad et al. 2025 (DocEng):** a *default* (non-fine-tuned) model gave **no** time savings over manual transcription in the correct-don't-transcribe loop; only a *fine-tuned* model did. Our v2 shipped model at CER 77.75% is effectively a default model for new plates. Expect the first correction round to be ~as slow as transcribing from scratch; the speedup (if any) comes in round 2+ after Stage 4 produces a better model. Do not budget scholar-hours assuming round-1 speedup.

---

## Stage 1 — Preprocessing (wire what exists)

Modules exist standalone, never chained:

| Module | Status | Gap |
|---|---|---|
| `preprocessing/preprocessor.py` | Whole-page denoise/contrast/binarize/deskew/normalize | Uses whole-page deskew, not per-fragment |
| `preprocessing/binarize.py` | Sauvola + nlbin, bleed-through detection | Produces geometry masks only, not BLLA input masks |
| `preprocessing/deskew.py` | Per-fragment Hough deskew from mask | **Not wired** — `preprocessor.py` still uses whole-page |
| `segmentation/fragment_isolation.py` | CC-based fragment isolation | Standalone |
| `segmentation/row_bands.py` | Row-band crop from CC clustering | Standalone, used by annotation autosuggest only |
| `segmentation/line_extraction.py` | Line crop extraction | Standalone |

### v8 pipeline (wire, don't rewrite)

```
plate.png
  → fragment_isolation.isolate_fragments()        # CC clustering → fragments
  → for each fragment:
      deskew_fragment()                             # per-fragment Hough deskew
      binarize_for_geometry()                       # Sauvola/nlbin for geometry
  → manuscript_area.detect()                        # main text area
  → row_bands.extract()                              # row-band crops (annotator UI)
  → [annotator commits per-fragment baselines per §4 convention]
  → BLLA fine-tune (Stage 3) / recognition fine-tune (Stage 4)
```

- **Single integration point:** a `msocr/preprocessing/pipeline.py` that chains the existing modules in order. ~50 lines. No new algorithms.
- **Output:** per-plate `_fragments/` directory with isolated+deskewed fragments + a geometry binary mask for BLLA input.
- **Ponytail:** the modules exist. The wiring is the work. Do not refactor the modules.

---

## Stage 2 — Annotation policy (implement, don't just document)

Covered by Stage 0.3 (backend/export) and `fix-a-v9-annotation-plan.md` §8-9 (frontend). The policy is in `docs/kraken-fragmentary-manuscripts.md` §4 + `docs/ANNOTATION.md`; neither the export code nor the frontend follows it. v8 makes the code match the docs.

### 2.1 Within-line uncertainty
- Leiden underdot (`U+0323`) for partially legible glyphs — already in our convention, ensure annotators use it.
- `�` (`U+FFFD`) for "there was a glyph here, I have no idea what" — keep as-is.
- Lost stretch of known length: **split the line.** Do not encode `[...]` in GT (Kraken has no validated lacuna token; `Mind the Gap`/TrCroT failed to bracket reliably — lib-1, Claim 4).

### 2.2 DamageZone for lacunae
- Annotator draws `DamageZone` polygon over the lacuna.
- `ketos segtrain --suppress-regions` doesn't suppress *damage* — it suppresses *region classes*. To prevent BLLA from looking for baselines in a damage zone, use the page-level `segment(mask=…)` at inference time (Stage 5) and avoid drawing baselines there at annotation time. There is no training-time per-pixel ignore channel (lib-1, Claim 6).
- **Frontend:** DamageZone is already in the region palette but hidden behind an "Advanced" toggle in `AnnotateEditor.tsx`. Promote to default visibility for fragmentary plates. (XS fix, see annotation plan §8.)

### 2.3 Bootstrap dump-preds pipeline (feeds the correct-don't-transcribe loop)

`scripts/dump_c2av12_preds.py` is a one-off research script — hardcoded to `c2av12_page_12.xml` and one model, no CLI args, not integrated with the annotation UI. The correct-don't-transcribe loop (annotation plan §5) needs a generalized "run current model over N unannotated plates, dump predictions + confidence into the annotation session" command.

- **Generalize** `dump_c2av12_preds.py` into a `msocr dump-preds` subcommand: take model path, XML glob or plate-id list, output dir; write predictions + per-line confidence into the v2 annotation state's `transcript` + a new `confidence` field.
- **Wire** into `/autosuggest` (or a new `/bootstrap` endpoint) so the frontend can trigger it for a fresh plate rather than the annotator running a separate CLI.
- **Ponytail:** the script exists; generalize, don't rewrite. ~50 lines.

---

## Stage 3 — Segmentation fine-tuning (BLLA with `--suppress-regions`)

Confirmed: `--suppress-regions` is the right knob (lib-1, Claim 1). Fine-tune baselines on Sogdian without a region policy.

### 3.1 Recipe

```bash
ketos segtrain -o models/kraken/blla_sogdian_finetune.mlmodel \
  --suppress-regions \
  --valid-baselines Default \
  --bounding-regions MainZone \
  --augment \
  -r 5e-5 --lag 10 --min-epochs 20 --quit early \
  dataset/christian_sogdian_c2av/gt/*.xml
```

- Base model: `blla.mlmodel` (Kraken default). Fine-tune, don't train from scratch.
- `--suppress-regions`: baseline-only, ignore region classes (lib-1, Claim 1).
- `--valid-baselines Default`: exclude `Heading`/`Interlinear` noise from `Default`-line training.
- `--bounding-regions MainZone`: prevent polygons straying across column edges.
- `--augment`: albumentations line augmentation. Neutral for lacunae (no built-in hole simulator) but helps small data.
- **Gate:** annotate ≥30 fragment-aware plates before this stage. BLLA fine-tuning on 12 plates is below the documented floor for seg fine-tuning (`kraken-training-data-research.md` recommends 100–300 pages). If we stay at 12 plates, seg fine-tuning may not help — measure with `ketos segtest` on a held-out plate.

### 3.2 What this does NOT solve

- The default BLLA's three weak spots (non-Latin, diacritized, fragmentary) are real (lib-1, Claim 2), but fine-tuning on 12 plates may not move them. The remedy in Kraken literature is "fine-tune on a couple of pages" (maintainer on issue #745), but that's for a couple of out-of-distribution pages, not 12 fragmentary ones.
- **Orli** (lib-1, Claim 7) is now published (arXiv 2606.04166, github.com/mittagessen/orli) — ConvNeXtV2 + transformer decoder, autoregressive Bézier baselines, joint reading order. Not in Kraken mainline. **Open risk:** evaluate Orli on our fragmentary plates as a side experiment; if it handles fragmentary baselines better than BLLA-finetuned, it could replace Stage 3 entirely. Low effort to try (code + weights available), high potential payoff.

---

## Stage 4 — Recognition fine-tuning (`--resize new`, not `union`)

**v8 change from v7:** switch `--resize union` → `--resize new`. The Kraken docs explicitly warn: "When fine-tuning, it is recommended to use **new** mode not **union** as the network will rapidly unlearn missing labels in the new dataset" (lib-1, Claim 5). v4's 71.43%→80.09% regression is the documented failure mode of `union`.

### 4.1 Why `new` over `union`

- `union` keeps old output neurons for characters absent from the new training data. Those neurons receive zero gradient, drift, and destabilize the shared backbone. This is exactly the v4 failure.
- `new` rebuilds the classifier head for exactly the Sogdian alphabet. The backbone is preserved (with `--freeze-backbone`), only the head learns from scratch.
- We don't need a multi-script model for the runtime path — `run_htr_service` already resolves one model per language. A Sogdian-only recognizer is fine.
- **When `union` would be justified:** if we wanted a single model recognizing both Syriac and Sogdian national script (U+10F30). We don't — `multi-script-htr-research.md` says separate models per script. Use `new`.

### 4.2 Recipe

```bash
ketos train -o models/kraken/c2av_finetune_v8 \
  -i models/kraken/sophro_mhiro_syriac.safetensors \
  --resize new \
  --freeze-backbone 999999 \
  --warmup 200 \
  --augment \
  -r 5e-5 --lag 10 --min-epochs 20 --quit early \
  --base-dir R \
  -f xml dataset/christian_sogdian_c2av/gt/*.xml
```

- `--resize new`: rebuild classifier for Sogdian alphabet (replaces v7's `union`).
- `--freeze-backbone 999999`: frozen-forever regime. v2 found this is the only stable regime at 148 lines (`fix-a-v2-syriac-script-finetune.md`). v4 confirmed unfreezing regresses.
- `--warmup 200`: linear warmup over 200 samples. Kraken docs recommend 1–2 epochs of warmup with `--freeze-backbone` (lib-1, Claim 5). At 148 lines / batch, 200 samples ≈ 1.3 epochs.
- `--augment`: line augmentation (use `line_augment.py` if kraken 7.x's built-in is insufficient — `line_augment.py` is ported from kraken's removed linegen).
- Base model: `sophro_mhiro_syriac.safetensors` (Estrangelo/Eastern Syriac, script-adjacent). Per `multi-script-htr-research.md`, Christian Sogdian uses the Syriac block (U+0710), so a Syriac base is correct.
- **MS Jer 36 intermediate (v6 Stage 1):** if Stage 0.1 unblocks it, use the MS Jer 36-adapted model as the base instead of raw Sophro Mhiro. If it stays blocked, skip — Sophro Mhiro alone got v2 to CER 77.75%.

### 4.3 Gate

- **Run LOOCV (Stage 0.2) first.** Do not start Stage 4 until the 12-fold LOOCV report exists for the current shipped model.
- After Stage 4, re-run LOOCV on the v8 model. Compare per-fold. If v8 doesn't beat v2's frozen-forever regime on mean CER across 12 folds, revert.

---

## Stage 5 — Inference pipeline (wire runtime to fragments)

`run_htr_service` in `service/runtime.py` ignores `script_variant` (exp-1, finding 5) and has no post-correction hook. `save_output` ignores `image_paths`/`original_pdf_path`/`language` (exp-1, finding 6 — dead code).

### 5.1 Wire the fragment mask into inference

- Per-plate `mask.png` over known damage bands. Pass `-m mask.png` to `kraken segment` in the inference path.
- This is an inference-time-only mask (lib-1, Claim 6). It stops BLLA from *finding* lines in a zone. It does not help a line that legitimately crosses a gap (we handle that by splitting baselines at annotation time, §2.2).

### 5.2 Wire the v8 model into runtime resolution

- `DEFAULT_HTR_MODELS["sogdian"]` in `runtime.py` currently points to `models/kraken/sogdian_manuscript.mlmodel` (which doesn't exist on disk). Point it at the v8 model from Stage 4.
- Honor `script_variant` — if `christian-syriac-script`, resolve to the v8 model; if `sogdian-national` (future), resolve to a separate U+10F30 model. Don't conflate.

### 5.3 Remove dead params or wire them

- `save_output`'s `image_paths`/`original_pdf_path`/`language` are accepted and ignored. Either wire them (write per-image metadata into JSON output) or delete them from the signature. Ponytail: delete unless a consumer needs them.

---

## Stage 6 — Post-OCR correction (non-neural, NOT ByT5)

**v8 change from v7:** drop ByT5-small. Replace with a non-neural baseline.

### 6.1 Why not ByT5

- Every published ByT5 post-correction success uses 500–10,000+ training pairs (lib-1, Claim 3). Momtaz et al. 2025 (smallest documented case) used <500 line samples with ByT5-small on Latin incunabula and got ~15% CER. We have 150.
- The Arabic-Turkic benchmark found a train-derived char-confusion baseline (edit-distance + confusion matrix) achieved CER 0.0825 vs ByT5-small's 0.0799 — a negligible gap. For 150 pairs, the non-neural baseline is almost certainly superior.
- Overfitting/memorization risk on a 300M-parameter model with 150 pairs is extreme.
- Synthetic data augmentation (Guan & Greene 2024, Method 4) could stretch 150 pairs to ~600, but that's still below documented minimums.

### 6.2 v8 post-correction: weighted-Levenshtein + char-confusion matrix

- **Train** a character-confusion matrix from the LOOCV (Stage 0.2) predictions vs GT. For each (predicted, gold) char pair, count co-occurrences within edit-distance-1.
- **Apply** weighted-Levenshtein at inference: the cost of substituting char `a` for char `b` is inversely proportional to the confusion probability `P(b|a)` from the matrix.
- **Lexicon gate:** the existing lexicon post-corrector (`scripts/lexicon_postcorrect.py`) **failed** at CER 79.90% → 85.57% (`fix-a-v3-syriac-script-finetune.md`). Re-try only when holdout CER drops below ~50%. Deletions are the dominant error mode (83.4% per v3 confusion audit) — a lexicon can't insert missing chars, so it can't help until substitutions dominate.
- **Implementation:** ~100 lines in a new `msocr/postcorrection/char_confusion.py`. No new deps (stdlib + numpy, already installed). Ponytail: no transformers, no torch, no huggingface.

### 6.3 What this cannot do

- It cannot insert missing characters (deletions are 83.4% of errors). Only a better recognizer (Stage 4) or more data (Stage 0.4) fixes deletions.
- It cannot fix the CTC alignment collapse (v3's finding). That's a recognizer problem, not a post-correction problem.

---

## Stage 7 — Tool matrix (updated)

| Stage | Tool | Status | v8 action |
|---|---|---|---|
| 0.1 | RunPod runner | Blocked (ssh_exec bug) | Fix the one method or kill pod |
| 0.2 | `build_lopo_manifests.py` + `harness.py` | 1/12 folds done | Run all 12, aggregate |
| 0.3 | `annotation_api.py` PAGE XML export | Missing fragment convention | Add DamageZone, RTL ReadingOrder, per-fragment baselines |
| 1 | `preprocessing/*` modules | Standalone | Wire into `pipeline.py` |
| 2 | `ANNOTATION.md` | Page-level only | Add fragment convention |
| 3 | `ketos segtrain` | `--suppress-regions` confirmed | Fine-tune BLLA, gate on ≥30 annotated plates |
| 3 (alt) | Orli | Published, not in Kraken mainline | Evaluate as side experiment |
| 4 | `ketos train` | `--resize new` (was `union`) | Switch mode, add `--warmup`, gate on LOOCV |
| 5 | `runtime.py` | Ignores `script_variant`, dead params | Wire v8 model, honor variant, remove dead params |
| 6 | ByT5 post-correction | DROPPED (150 pairs too few) | Replace with weighted-Levenshtein + char-confusion |
| 6 (lexicon) | `lexicon_postcorrect.py` | Failed at 79.90% CER | Re-try only if CER < 50% |
| 7 | `line_augment.py` | Exists, not wired | Wire into Stage 4 `--augment` if needed |

---

## Open risks (v8)

1. **Annotation scale.** 87/99 plates unannotated. Every model claim is hedged by a 178-line floor. The single highest-leverage action is annotating more plates (Stage 0.4). Without it, v8 optimizes within a regime where CER CIs are ±4–7 pts.
2. **Annotation protocol consistency.** Couture et al. 2023 found inconsistent transcription conventions between plates/annotators silently corrupt small datasets. Writing the graphemic convention down once and applying it by checklist is a prerequisite for any "annotate more plates" effort — more plates with inconsistent conventions is worse than fewer plates done consistently. (See annotation plan §1.)
3. **Bootstrap round-1 speedup is not guaranteed.** Mohammad et al. 2025 measured a *default* model giving no time savings over manual transcription; only a *fine-tuned* model did. Our v2 (77.75% CER) is effectively a default model for new plates. Do not budget scholar-hours assuming round-1 of the correct-don't-transcribe loop is faster than blind transcription. (See annotation plan §5.)
4. **Orli.** Published, not in Kraken mainline. Could replace Stage 3 if it handles fragmentary baselines better. Low effort to evaluate (code + weights available). Do this as a side experiment, not on the critical path.
5. **`--resize new` is untested on our data.** v2/v3/v4 all used `union`. Switching to `new` is the docs-recommended mode but we have no local evidence it beats `union` frozen-forever at 148 lines. LOOCV (Stage 0.2) on the `new` model vs the `union` model is the test.
6. **LOOCV cost.** 12 folds × ~30 min/fold on a single GPU = ~6 hours. Cheap. No excuse for not having it.
7. **v6 Stage 1 (MS Jer 36).** Blocked. If we defer it, we lose the intermediate domain adaptation step. If we fix it, we get a better base for Stage 4. Fix the `ssh_exec` bug (one method) or kill the pod.
8. **Deletions dominate.** 83.4% of errors are deletions (v3). No post-correction fixes deletions. Only a better recognizer + more data does. v8 does not pretend otherwise.
9. **CurT/Orli fragmentary performance is untested.** The Orli paper demonstrates zero-shot generalization and fine-tuning adaptability but does not benchmark fragmentary material specifically. Treat the "Orli handles fragments better" hypothesis as untested until we run it on our plates.
10. **Graphemic-transcription claim is advocated, not empirically proven for non-Latin.** CATMuS (Latin-script, ICDAR 2024) advocates graphemic transcription as a standardization choice; Couture 2023 found graphetic→graphemic switching improved results on French handwriting. No study tests this for Syriac-tradition scripts. Our g/γ→GAMAL collapse and t/θ qushshaya/rukkakha distinction are well-motivated graphemic choices, but the claim that graphemic *outperforms* diplomatic/normalized is extrapolation, not direct evidence for our domain. (See annotation plan §1.)
11. **Active-learning-at-150-lines is unvalidated.** Published active-learning HTR work (Tsakalis 2021, VUT 2024) operates on datasets 10–100× larger than 178 lines. At our scale, confidence estimates may be too noisy for uncertainty sampling to help; rare-class targeting is the safer first lever. (See annotation plan §6.)

---

## What v8 deliberately does NOT do

- **No ByT5 / no transformer post-correction.** 150 pairs is below the documented floor. Non-neural baseline only.
- **No `--resize union` for fine-tuning.** Docs warn against it; v4 failed with it. `new` mode.
- **No lacuna token in GT.** Kraken has no validated convention; `Mind the Gap` failed to bracket reliably. Split the line.
- **No per-pixel ignore channel.** Doesn't exist in Kraken (lib-1, Claim 6). Use `--suppress-regions`, `--bounding-regions`, and inference-time `segment(mask=…)` instead.
- **No multi-script joint model.** Separate models per script (`multi-script-htr-research.md`).
- **No beam search.** Unavailable in kraken 7.0 (`fix-a-v3-syriac-script-finetune.md`). Greedy decoding only.
- **No new dependency for post-correction.** Stdlib + numpy. No transformers, no torch, no huggingface for Stage 6.

---

## Sources (new in v8)

- Kraken `ketos segtrain` docs: [kraken.re/6.0.0/training/segtrain.html](https://kraken.re/6.0.0/training/segtrain.html)
- Kraken `ketos train` docs (`--resize new` vs `union`): [kraken.re/5.1/ketos.html](https://kraken.re/5.1/ketos.html)
- BLLA model card: [zenodo.org/records/14602569](https://zenodo.org/records/14602569)
- Kraken issues: [#745](https://github.com/mittagessen/kraken/issues/745), [#677](https://github.com/mittagessen/kraken/issues/677), [#773](https://github.com/mittagessen/kraken/issues/773)
- Orli paper: [arXiv 2606.04166](https://arxiv.org/abs/2606.04166) · code: [github.com/mittagessen/orli](https://github.com/mittagessen/orli)
- Momtaz et al. 2025 (ByT5 on Latin incunabula, <500 lines): *Modular Pipeline for Text Recognition in Early Printed Books Using Kraken and ByT5*, Electronics 14(15):3083
- Guan & Greene 2024 (synthetic data for post-OCR): [arXiv 2408.02253](https://arxiv.org/abs/2408.02253)
- Arabic-Turkic OCR benchmark (non-neural vs ByT5): [github.com/Therad445/low-resource-arabic-script-turkic-ocr](https://github.com/Therad445/low-resource-arabic-script-turkic-ocr)
- Lacuna Reconstruction (Islamicate, self-supervised pretraining): [aclanthology.org/2022.findings-naacl.15](https://aclanthology.org/2022.findings-naacl.15/)
- Mind the Gap / TrCroT (lacuna token attempt): [arXiv 2407.00250](https://arxiv.org/abs/2407.00250)
- Dead Sea Scrolls fragment segmentation: [arXiv 2406.15692](https://arxiv.org/html/2406.15692)
- CATMuS Medieval (graphemic transcription standard, Latin-script ICDAR 2024): [hal-04453952](https://inria.hal.science/hal-04453952) · [catmus-guidelines.github.io](https://catmus-guidelines.github.io/)
- Couture et al. 2023 (real-world HTR training, protocol consistency, `<unclear>` line-exclusion): [JDMDH 10.46298/jdmdh.10542](https://doi.org/10.46298/jdmdh.10542)
- Mohammad et al. 2025 (assisted-transcription time measurement, default-model no-speedup): [DocEng 10.1145/3704268.3748677](https://doi.org/10.1145/3704268.3748677)
- Tsakalis 2021 (active learning for historical HTR): [Cambridge MPhil thesis](https://www.mlmi.eng.cam.ac.uk/files/konstantinos_tsakalis_8224911_assignsubmission_file_dissertation_signed.pdf)
- HTR Winter School 2024 Syriac (ÖNB Cod. Syr. 1, Serto, 2869 lines, CC-BY): [Zenodo 10.5281/zenodo.14714089](https://doi.org/10.5281/zenodo.14714089)
- HTR Winter School 2025 Syriac (MS Jerusalem Saint Mark's 36, Estrangelo): [Zenodo 10.5281/zenodo.18157525](https://doi.org/10.5281/zenodo.18157525)
- Tarride et al. 2023 (crowdsourced HTR annotation guidelines, exclude-illegible standard): [DocEng 10.1145/3604951.3605517](https://doi.org/10.1145/3604951.3605517)

Prior plans in this series: v1 (failed, `--resize union` unfrozen), v2 (shipped, frozen-forever, CER 77.75%), v3 (shipped, 200-epoch CER 71.43%, lexicon failed), v4 (failed, freeze-rows regressed 71.43%→80.09%), v5 (plan, LOOCV first), v6 (plan, MS Jer 36 intermediate, blocked), v7 (plan, ByT5 + `union`, superseded by this doc).

Companion document: [`fix-a-v9-annotation-plan.md`](./fix-a-v9-annotation-plan.md) — annotation methodology + eScriptorium-parity frontend work list.