# msocr Per-Style-Group HTR Training Pipeline — Design

- **Status:** Approved (2026-06-17)
- **Source:** `docs/Instruction.md` (plan request)
- **Author:** Orchestrator, from brainstorming session
- **Supersedes:** `AGENTS.md` / `README.md` / `CONTRIBUTING.md` "Removed historical scope: RunPod submission, multi-stage orchestration" lines — those scopes are explicitly re-opened by this design.

## 1. Goal

Build a per-style-group Sogdian/Syriac HTR training, annotation, and evaluation pipeline for the Berlin Turfan collection, with training executed on RunPod GPU Cloud Pods and a custom annotation UI inside the `msocr` repo.

Two parallel script-block models from the start:

- **Sogdian block (U+10F30)** — Manichaean + Buddhist Sogdian, fine-tuned from the OpenITI Arabic base (closest RTL + cursive joining behaviour).
- **Syriac block (U+0710)** — Jingjiao / Christian Sogdian, fine-tuned from a Syriac base.

Models are keyed by **style group** (script + period + scribal style), not by single manuscript, per Kraken maintainer guidance (issue #711: 100 pages from 100 documents beats 100 pages from 1 document — single-manuscript models overfit).

## 2. Locked decisions (from clarifying questions)

| # | Decision | Why |
|---|---|---|
| D1 | Reopen RunPod submission + minimal procedural orchestration scope in guardrail docs | User explicitly re-opens the previously-removed scope. |
| D2 | Per-style-group models, pooled manuscripts (not per-single-manuscript) | Matches Kraken maintainer guidance; avoids overfit. |
| D3 | Custom annotation UI inside `msocr` — no eScriptorium, no Astro | User refuses eScriptorium (hard constraint); Astro is static-only and wrong for an input UI. |
| D4 | FastAPI + HTMX + Alpine.js on the existing `annotation_api.py` | Better than Gradio (which is demo-grade), no JS build, no new heavy dep, serves the plan-as-HTML on the same app. |
| D5 | Two parallel script-block models (Sogdian U+10F30 + Syriac U+0710) from the start | Berlin Turfan Christian/Jingjiao material is largely Syriac-script; Manichaean/Buddhist is Sogdian-script. |
| D6 | Thin evaluation: wrap `ketos test` + aggregate per-manuscript / per-style-group, JSON + Markdown report | Reuses Kraken's own metrics (CER, WER, case-insensitive CER, per-character accuracy, char-confusion). No invented metrics. Ponytail rung 4. |
| D7 | RunPod GPU Cloud Pods via `runpod` SDK (SSH-style), not Serverless Endpoints | Pods fit 1–2 hr fine-tunes (<$1 each on a 4090), have persistent disk for resume. Serverless has timeouts that make it wrong for batch training. |
| D8 | Minimal procedural orchestrator (Approach A), not a job queue (B) or workflow engine (C) | Matches the actual scale (2 parallel models now, ~2–10 total foreseeable). Procedural walker: one style-group → runner → eval. Durable queue / DAG engine is an additive upgrade if needed later. |

## 3. Architecture

### 3.1 Guardrail docs reopened (must edit in the same commit as the first RunPod code)

| File | Current text | New text |
|---|---|---|
| `AGENTS.md` "Key Conventions" | "HTR-only scope. Do not reintroduce printed OCR routing, Tesseract fallbacks, RunPod submission, HAR promotion, or multi-stage orchestration." | "HTR scope with remote training. RunPod GPU Cloud Pod submission is supported for `ketos train` fine-tuning via `msocr train-remote`. Multi-stage orchestration is a minimal procedural walker (one style-group at a time), not a DAG engine. Tesseract/OCRmyPDF/printed-OCR/HAR remain out of scope." |
| `README.md` "Removed scope" | Lists RunPod submission + multi-stage orchestration as removed. | Move RunPod to "Active scope (remote training)"; keep Tesseract/OCRmyPDF/printed OCR/HAR as removed. |
| `CONTRIBUTING.md` "HTR-Only Scope" | "Do not reintroduce … RunPod submission … multi-stage orchestration" | Mirror AGENTS.md wording. |
| `docs/Instruction.md` line 30 | "Removed historical scope: … RunPod submission … multi-stage orchestration" | Replace with "Active scope: RunPod GPU Cloud Pod training via `msocr train-remote`; procedural per-style-group orchestration." Also commit the currently-untracked `docs/Instruction.md` as the plan source. |

### 3.2 New module map (additive — no existing module deleted)

```
msocr/
├── cli.py                          # + train-remote, evaluate, annotate (new subcommands)
├── language_registry.py            # + script_block field (U+10F30 vs U+0710)
├── models/
│   └── inference.py                 # unchanged (already correct for RTL)
├── service/
│   ├── api.py                       # unchanged
│   ├── gradio_demo.py               # unchanged (kept for quick HTR demo)
│   ├── runtime.py                   # unchanged (local HTR runtime, intentionally local-only)
│   ├── deploy.py                    # unchanged
│   ├── annotation_api.py            # + /ui route (Jinja2+HTMX+Alpine.js), /plan route (HTML plan)
│   └── annotation_ui/               # NEW: templates + static assets for the annotation UI
│       ├── templates/line.html.j2  # per-line image + RTL textbox + Save/Next
│       ├── templates/page.html.j2   # page overview + current-line highlight
│       ├── templates/plan.html.j2   # renders the design doc as HTML
│       ├── static/alpine.min.js     # vendored, no CDN dep
│       ├── static/htmx.min.js       # vendored
│       └── static/sogdian_keyboard.html  # on-screen character palette fragment
├── evaluation/
│   ├── metrics.py                  # unchanged (cer/wer primitives kept)
│   └── harness.py                   # NEW: wraps ketos test, aggregates per-manuscript/per-style-group
├── training/
│   ├── ketos_trainer.py             # REWRITE for Kraken 7.0 CLI shapes (global flags, .safetensors, --load/--resize/--freeze-backbone)
│   ├── runpod_runner.py             # NEW: submit pod, SSH ketos train, poll, download artifact
│   └── orchestrator.py              # NEW: walk a style_group in a manifest -> runner -> eval
├── data/
│   ├── manifest.py                  # + script_block field, + style_group_id key
│   └── manifests/                   # NEW: actual JSON manifests for Berlin Turfan Sogdian + Syriac
└── output/
    └── formats.py                   # + benchmark report writer (JSON + Markdown)
```

### 3.3 New deps

| Dep | Version | Released | Python | Why | Adds to runtime? |
|---|---|---|---|---|---|
| `runpod` | `1.9.1` | 2026-06-01 | `>=3.8` (3.12 OK) | GPU Cloud Pod submission via SDK | yes |
| `paramiko` | latest | — | — | SSH into RunPod pods to exec `ketos train` | yes |
| `jinja2` | transitive (FastAPI/Starlette) | — | — | annotation UI templates | no (already present) |
| `python-multipart` | already present | — | — | `annotation_api` file upload | no |

**Not added:** `astro`, `react`, `vite`, `redis`, `rq`, `celery`, `prefect`, `airflow`, `datasets`, `huggingface_hub`. No JS toolchain, no queue, no workflow engine.

### 3.4 What stays unchanged (explicit non-goals)

- `msocr/models/inference.py` — already correct for Sogdian RTL (`horizontal-rl`), Kraken 7.0 `models.load_any` still works. Migration to `SegmentationTaskModel` is a later, separate task.
- `msocr/service/runtime.py` — local HTR runtime, intentionally local-only. RunPod is for *training*, not inference. Inference stays local + the existing env-var model resolution.
- `msocr/service/gradio_demo.py` — kept for the quick HTR demo. Not used for annotation.
- The default `models/kraken/sogdian_manuscript.mlmodel` path convention — kept. After training, the downloaded `.safetensors` lands at the per-style-group path the manifest names.

### 3.5 Kraken 7.0 migration (required, blocking)

`msocr/training/ketos_trainer.py` is out of sync with Kraken 7.0.2 (released 2026-05-04). Must be rewritten before any training runs:

| Issue in current `ketos_trainer.py` | Kraken 7.0 reality |
|---|---|
| `--workers`, `--device`, `--precision` passed to `train` subcommand | Global on main `ketos` since 6.0: `ketos -d cuda:0 --workers 8 train ...` |
| `--partition str(validation_split)` | Not a valid `ketos train` flag — partitions are defined in the manifest, not CLI |
| No `--load` / `--resize` / `--freeze-backbone` | Required for fine-tuning — the wrapper can only train from scratch today |
| Output `.mlmodel` (assumed by `get_available_models`, `inference.py:230`) | 7.0 default is `.safetensors`; `.mlmodel` is legacy CoreML |
| No YAML experiment config support | 7.0 supports `ketos train --config experiment.yaml` for reproducibility |

Either pin `kraken<7` in `pyproject.toml` **or** rewrite `ketos_trainer.py` to 7.0 CLI shapes. The design picks **rewrite** (the existing wrapper is the only Kraken CLI integration point, and 7.0 is the supported line).

## 4. Data flow

### 4.1 Annotation → training data

```
manuscript folio image (PNG/TIF)
    │
    ▼
msocr preprocess -i <dir>          # binarize, deskew (existing CLI)
    │
    ▼
annotation_api session              # POST /api/sessions (existing endpoint)
    │
    ▼
msocr annotate                      # NEW CLI: opens the /ui route in a browser
    │                                # line-by-line: image + RTL textbox + Save/Next
    │                                # Sogdian character palette for U+10F30 input
    │                                # Syriac character palette for U+0710 input
    │
    ▼
GET /api/sessions/{id}/export?format=page   # existing endpoint, exports PageXML
    │
    ▼
ketos compile -f page -o train.arrow *.xml   # Kraken 7.0 compile step
    │
    ▼
training manifest JSON              # data/manifests/berlin-turfan-{sogdian,syriac}-v1.json
```

### 4.2 Training (RunPod GPU Cloud Pod)

```
msocr train-remote \
    --manifest berlin-turfan-sogdian-v1 \
    --style-group manichaean-early \
    --base-model openiti-arabic-base \
    --device cuda:0 \
    --pod-gpu "RTX 4090" \
    --pod-image msocr-kraken7:latest \
    --epochs 50 --min-epochs 20 --lag 10 --augment \
    --freeze-backbone 5000 --resize union
    │
    ▼
orchestrator.walk_style_group(manifest, style_group_id)
    │
    ▼
runpod_runner.submit_pod(...)       # runpod.create_pod(name, image, gpu_type)
    │
    ▼
runpod_runner.ssh_ketos_train(...)   # paramiko SSH, exec:
    │                                #   ketos -d cuda:0 --workers 8 train \
    │                                #     --load base.safetensors \
    │                                #     --resize union --freeze-backbone 5000 \
    │                                #     --augment --min-epochs 50 \
    │                                #     -f binary -t train.arrow -e val.arrow \
    │                                #     -o /workspace/models/<style_group>
    │
    ▼
runpod_runner.poll_until_done(...)  # runpod.get_pod(id) until EXITED
    │
    ▼
runpod_runner.download_artifact(...) # scp the .safetensors + .ckpt to local models/kraken/
    │
    ▼
runpod_runner.terminate_pod(...)    # runpod.terminate_pod(id)
    │
    ▼
local: models/kraken/<style_group>.safetensors
```

### 4.3 Evaluation

```
msocr evaluate \
    --manifest berlin-turfan-sogdian-v1 \
    --style-group manichaean-early \
    --model models/kraken/manichaean-early.safetensors
    │
    ▼
harness.run_evaluation(manifest, style_group_id, model_path)
    │
    ├─ for manuscript_id in style_group.holdout:
    │     ketos test -m <model> -f binary <manuscript.arrow>  →  parse CER/WER
    │
    ├─ aggregate per-manuscript
    ├─ aggregate per-style-group
    │
    ▼
output/formats.write_benchmark_report(...)
    │
    ├─ reports/<manifest>__<style_group>__<model>.json    # machine-readable
    └─ reports/<manifest>__<style_group>__<model>.md      # human-readable table
```

### 4.4 Plan-as-HTML (the original Instruction.md ask)

```
msocr annotate   # or any msocr FastAPI app
    │
    ▼
GET /plan     # new route on annotation_api.py
    │
    ▼
renders docs/plans/2026-06-17-msocr-training-pipeline-design.md → HTML
    │
    ▼
served as a readable web page at http://127.0.0.1:8001/plan
```

Same FastAPI app as the annotation UI — no separate Astro site, no JS build.

## 5. Error handling

### 5.1 RunPod failure modes

| Failure | Detection | Recovery |
|---|---|---|
| Pod submission fails (quota, bad GPU type, bad image) | `runpod.create_pod` raises | Surface the SDK error to CLI; exit non-zero. No retry. |
| Pod starts but SSH unreachable | `paramiko.SSHException` on connect | Retry SSH up to 3 times with 30s backoff (pod boot can be slow). Then surface. |
| `ketos train` exits non-zero on the pod | SSH `exec_command` returns non-zero exit status | Pull the last 200 lines of stderr from the pod, surface to CLI, terminate pod. |
| Pod dies mid-training (GPU OOM, host lost) | `runpod.get_pod` returns `EXITED` before `poll_until_done` expected it | Surface, offer `--resume` using the pod's persistent disk checkpoint (`ketos train --resume checkpoint_0005.ckpt`). |
| Artifact download fails | `scp` raises | Retry download up to 3 times, then surface. Pod is **not** terminated on download failure (so the artifact survives for manual recovery). |
| RunPod API key missing/invalid | `runpod` SDK raises `AuthenticationError` | Surface with "set RUNPOD_API_KEY env var" message. |
| Network flake during polling | `runpod.get_pod` raises transient | Retry up to 5 times with 60s backoff. |

**Principle:** never silently lose a training artifact. If the pod ran `ketos train` successfully, the `.safetensors` on the pod's persistent disk must be retrievable until the user explicitly terminates.

### 5.2 Kraken training failure modes (Sogdian/Syriac specific)

| Failure | Detection | Recovery |
|---|---|---|
| CER stuck at ~0.30 after 2 epochs | `ketos test` on validation partition | Kraken tutorial lists "non-reordered right-to-left" as a cause. Check `text_direction="horizontal-rl"` is set on segmentation and `--base-text-direction horizontal-rl` on `ketos test`. Abort, fix, restart. |
| Fine-tuning collapse ("best model outputs same string for all lines") | `ketos test` CER ≈ 1.0 or all predictions identical | Issue #711 fix: `--augment --min-epochs 50 --freeze-backbone 5000 --resize new`. |
| `Line polygon outside of image bounds` warnings | Kraken log during `ketos compile` | Validate baseline coordinates against image dimensions before compile. Skip out-of-bounds lines with a warning count. |
| Binarization mode mismatch ("trained on mode 1, got mode L") | Kraken log during training | Be consistent: `inference.py:66` always runs `binarization.nlbin`. Training data prep must match. |
| Combining marks (Sogdian U+10F46–U+10F50) dominate errors | Per-character accuracy in `ketos test` output | More training data for dotted forms; consistent dot placement in transcription guidelines. |
| Wrong Unicode block (Syriac folios in a Sogdian manifest or vice versa) | Manifest `script_block` field vs `language_registry` check at manifest load | `data/manifest.py` rejects the manifest with a clear error. |

### 5.3 Local fallback

`msocr train` (existing CLI, local `ketos_trainer.py`) stays as a fallback for offline / no-RUNPOD_API_KEY / debugging. The RunPod runner is the remote path; the local trainer is the debug path. Both produce `.safetensors` to the same `models/kraken/` directory.

## 6. Testing

### 6.1 Unit tests (no network, no GPU)

| Module | Test file | What |
|---|---|---|
| `data/manifest.py` | `tests/data/test_manifest.py` (extend) | `script_block` field is required and validated against `language_registry`; `style_group_id` resolves to a manuscript list; `manuscript_id` isolation across partitions still enforced. |
| `evaluation/harness.py` | `tests/evaluation/test_harness.py` (new) | Parses a fixture `ketos test` stdout JSON, aggregates per-manuscript and per-style-group correctly, writes the report files. |
| `evaluation/metrics.py` | `tests/evaluation/test_metrics.py` (new, currently empty dir) | `cer`/`wer` edge cases (empty ref, empty hyp, identical, RTL strings). |
| `training/ketos_trainer.py` | `tests/training/test_ketos_trainer.py` (new) | 7.0 CLI command shape is built correctly (global flags on `ketos`, `--load/--resize/--freeze-backbone` for fine-tune, `.safetensors` output path). `subprocess.run` is mocked. |
| `training/runpod_runner.py` | `tests/training/test_runpod_runner.py` (new) | `runpod` SDK is mocked; `paramiko` SSH is mocked. Verifies: pod submitted with right image+GPU, SSH command is the right `ketos` invocation, polling exits on `EXITED`, download copies to the right local path, pod terminated on success, pod **not** terminated on download failure. |
| `training/orchestrator.py` | `tests/training/test_orchestrator.py` (new) | Walks a fixture manifest's style_group, calls runner with right args, calls harness with the downloaded model, writes the report. All deps mocked. |
| `service/annotation_api.py` | `tests/service/test_annotation_api.py` (extend) | `/ui` route returns 200 + the line template; `/plan` route returns 200 + rendered HTML. |

### 6.2 Integration tests (no network, no GPU, real subprocess where safe)

| Test | What |
|---|---|
| `tests/integration/test_train_local_e2e.py` | Local `msocr train` on a tiny fixture manifest (3 lines), produces a `.safetensors`. Skipped if `ketos` not installed. |
| `tests/integration/test_evaluate_e2e.py` | `msocr evaluate` on a fixture manifest + a fixture model, produces JSON + Markdown report. |

### 6.3 Self-checks (ponytail: non-trivial logic leaves one runnable check)

Per ponytail, every non-trivial module gets one `__main__` self-check or one tiny test. The unit tests above cover this; no extra `demo()` functions needed.

### 6.4 What is NOT tested in CI

- Real RunPod submission (needs `RUNPOD_API_KEY`, costs money, slow). Documented in `tests/training/test_runpod_runner.py` with a skip marker and a manual runbook in `docs/runpod.md`.
- Real GPU training. Local `ketos train` on CPU is the integration fallback; real GPU is the RunPod path.
- Kraken 7.0 segmentation API migration. Out of scope for this design (inference stays on the legacy API, which still works in 7.0).

## 7. Open questions deferred to implementation

These are small enough to defer to the implementation plan, not blocking the design:

1. **Syriac base model choice.** The HTRMoPo repo has Syriac models via `ketos list`; pick the best one during implementation. Not a design-level decision.
2. **Sogdian character palette contents.** The 42 characters in U+10F30 + the 11 combining marks. Build the palette fragment during implementation; not a design decision.
3. **Docker image for RunPod pods.** `msocr-kraken7:latest` — build a `Dockerfile.train` during implementation. Base on `python:3.12-slim`, `uv sync`, install `kraken>=7.0.2`. Not a design decision.
4. **Manifest JSON files with real manuscript data.** The actual Berlin Turfan folio list is a data-collection task, not a design task. The manifest *schema* is fixed by this design; the *contents* are filled in during implementation.
5. **`docs/README.md` staged deletion.** Pre-existing (` D docs/README.md` in git status) — unrelated to this plan. Left alone.

## 8. Non-goals (explicit)

- Per-single-manuscript models (D2 — pooled style-group models instead).
- eScriptorium (D3 — refused by user).
- Astro web UI (D3 — wrong tool for input).
- Gradio as the annotation UI (D4 — demo-grade).
- RunPod Serverless Endpoints for training (D7 — wrong for batch training).
- Durable job queue / Celery / Prefect / Airflow (D8 — overkill for 2–10 models).
- Custom evaluation metrics (D6 — wrap `ketos test`, no invented metrics).
- Inference on RunPod (runtime stays local-only).
- Migration to Kraken 7.0 `SegmentationTaskModel` inference API (separate task; legacy API still works in 7.0).
- Printed OCR / Tesseract / OCRmyPDF / HAR promotion (remain out of scope per AGENTS.md).

## 9. Implementation order (hint for writing-plans)

Rough order, each step independently testable:

1. Edit guardrail docs (Section 3.1) + commit `docs/Instruction.md` + this design doc.
2. Rewrite `ketos_trainer.py` to Kraken 7.0 CLI shapes (Section 3.5).
3. Extend `data/manifest.py` with `script_block` + `style_group_id`; extend tests.
4. Add `evaluation/harness.py` + `output/formats.py` benchmark report writer + tests.
5. Add `service/annotation_api.py` `/ui` + `/plan` routes + `annotation_ui/` templates + tests.
6. Add `training/runpod_runner.py` + tests (mocked SDK + SSH).
7. Add `training/orchestrator.py` + tests.
8. Wire CLI: `msocr train-remote`, `msocr evaluate`, `msocr annotate`.
9. Build `Dockerfile.train` for RunPod pods.
10. Write `docs/runpod.md` runbook (API key, pod submission, manual recovery).
11. Create `data/manifests/berlin-turfan-sogdian-v1.json` + `berlin-turfan-syriac-v1.json` schemas (contents filled during data collection).

---

*Design approved 2026-06-17. Next step: invoke the `writing-plans` skill to produce the implementation plan.*