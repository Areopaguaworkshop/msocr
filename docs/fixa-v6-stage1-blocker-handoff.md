# fixa-v6 Stage-1 Blocker Handoff (2026-07-07)

> **STATUS UPDATE 2026-07-07 (v9):** Pod `phpyu59p2flw8g` **terminated** per
> Stage 0.1 decision in `docs/fix-a-v9-fragment-pipeline.md`. MS Jer 36
> intermediate domain adaptation is **deferred**. Pipeline Stage 4 will use
> the raw `sophro_mhiro_syriac.safetensors` base directly. To resume Stage 1
> later: fix the `ssh_exec` PTY bug (one method, see Blocker section), re-spin
> a pod, re-run `msocr train-remote --manifest data/manifests/ms-jer-36-adapt.json …`.
> All Stage-1 prereqs (downloaded models, manifest, polygon cache, namespace
> fixes) remain in place.

## Goal
Lower Christian Sogdian HTR CER via two-stage domain adaptation:
1. **Stage 1 (BLOCKED, pod killed, deferred to v9+)**: Domain-adapt Sophro Mhiro Syriac backbone on MS Jer 36 (~17,955 lines, Estrangelo/Eastern)
2. **Stage 2 (NOT STARTED)**: Freeze-backbone fine-tune 5 Sogdian-specific classes on c2av's 148 lines

## What is DONE

### Models & Data
- Downloaded + verified: Sophro Mhiro `syr_41transcribathon_docs_d_3.mlmodel` (Zenodo 17406773), MS Jer 36 (Zenodo 18157525, 133 bifolios, ~17,955 lines, PcGts 2013-07-15), Vienna GT (downgraded — wrong script family)
- Pre-converted safetensors: `models/kraken/sophro_mhiro_syriac.safetensors` — identical 36-char codec, same vgsl, same `seg_type: baselines` as the `.mlmodel`. Use as `--base-model` for stage 1
- Manifest built: `data/manifests/ms-jer-36-adapt.json` — 120 train / 8 val / 5 holdout, 133 unique ms_ids, style_group `jer36-adapt`, verified loads + enriches 135/135 polys on first case
- Stage-2 prereqs verified: `data/manifests/c2av-finetune.json` loads (10/1/1), `reports/c2av_union_codec.json` exists (37 base → 42 union classes, 5 new Sogdian rows), c2av XMLs+images resolve

### Code fixes (committed in working tree)
- `_enrich_xml_with_polygons` + `_resolve_image_for_xml` (in `msocr/data/manifest.py` or wherever they live) are now namespace-agnostic (detect from `root.tag`, support 2013+2019), with case-insensitive image fallback + sibling `images/` dir search. Verified on c2av (2019 ns, 17 lines, no regression) and Jer 36 (2013 ns, 135 lines, was broken now works)
- `runpod_runner.py:ssh_exec` — dropped `get_pty=True` (PTY never EOFs for quiet commands on this pod image). Now drains stdout+stderr channels separately with non-blocking `recv_ready()` and a deadline-based poll loop. **This fix is INCOMPLETE — see Blocker**
- Polygon cache added: `orchestrator.py:~218` — `~/.cache/msocr/poly_cache/` keyed by sha1 of `(xml_path, mtime, size, remote_img_name)`. Cache hit → copy to tmp (not rename, cross-device EXXD). Pre-populated with all 128 surviving enriched XMLs. Re-runs skip ~2.5h polygonization

### RunPod pipeline state
- `msocr train-remote` via `walk_style_group()` in `orchestrator.py`, `RunPodRunner`, SSH+`ketos train`+scp. Pod image `runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204` prebuilt. GPU RTX 3090. ~$0.60–$1 for adaptation
- RUNPOD_API_KEY in `.env` as `RunPod_API="<REDACTED: rotate if leaked>"` — must export as `RUNPOD_API_KEY` (name mismatch)
- ssh_exec bug PARTIALLY fixed (see Blocker). Pre-existing test failures in `tests/training/test_runpod_runner.py` are unrelated (test_ssh_endpoint_raises_when_no_public_port, test_run_training_full_lifecycle, etc. — verified via git stash)

### RunPod stage-1 attempt #3 state (THIS IS THE BLOCKER)
- Orchestrator PID 3779849 (`uv run msocr train-remote --manifest data/manifests/ms-jer-36-adapt.json --style-group jer36-adapt --base-model models/kraken/sophro_mhiro_syriac.safetensors --output-model models/kraken/sophro_jer36_adapted.safetensors --epochs 30 --warmup 200 --lr 1e-4 --augment --quit fixed --freeze-backbone 0`), log `/tmp/jer36_adapt3.log`
- Pod `phpyu59p2flw8g` SSH `174.94.157.109:46304` — **STILL RUNNING, all 260 files uploaded** (128 enriched XMLs + 128 JPGs + base.safetensors + train/val manifests), `/workspace/models/jer36-adapt` created
- Kraken is importable on pod (`pip install --quiet 'kraken>=7.0.2'` exited 0)
- **BUT**: orchestrator stuck at `running setup command: python3 -m pip install --quiet 'kraken>=7.0.2'` log line for ~40 min. No further log lines, no patches applied, no `ketos train` launched
- Pod-side: sshd session 3204 (orchestrator's) alive with NO child process (pip exited), but paramiko `exit_status_ready()` returns False forever. TCP connection ESTABLISHED (`...bell.ca:46304`), paramiko channel in zombie state

## The Blocker (root cause)

**`ssh_exec` poll loop deadlocks when sshd keeps the session channel open after the child process exits.**

Reproducible signature:
- Manual fresh paramiko SSH to same pod running same `pip install --quiet 'kraken>=7.0.2'` → exits in ~1s, `exit_status_ready()=True`, `recv_exit_status()=0` ✓
- Orchestrator's long-lived SSH connection (used for uploads then setup) → pip exits, sshd session has no child, but `exit_status_ready()` never fires, poll loop sleeps 0.2s forever

Hypothesis: paramiko's channel exit-status message is only delivered when the channel fully closes. With `get_pty=False`, after the child exits, sshd keeps the session channel open (perhaps waiting for the client to close). The orchestrator never reads EOF because `recv_ready()` is False (no buffered output) and `exit_status_ready()` returns False because the close handshake hasn't completed.

The fix that worked for the first wedge (drop `get_pty=True`) only addressed the PTY-never-EOF case. This is a SECOND failure mode: no-PTY + sshd-holds-channel-open.

## What is UNDONE

### Stage 1 — domain adaptation
- [ ] Fix `ssh_exec` watchdog: add idle timeout. If no exit_status in N seconds AND no channel output for M seconds, force-close channel and retry once. Or: switch from paramiko `exec_command` to `subprocess` + `ssh` CLI (the OS ssh client handles EOF/exit-status correctly). Or: explicitly call `chan.shutdown_write()` after exec to signal we won't send more input, which may prompt sshd to close
- [ ] Either: (a) kill stuck orchestrator, relaunch with fixed ssh_exec — pod will be recreated, all 260 files re-uploaded (~10 min), OR (b) kill orchestrator, manually drive the existing pod: apply 2 patches via direct SSH, launch `ketos train` in tmux, poll+download manually. Pod `phpyu59p2flw8g` is still up with all files — option (b) saves the upload time but bypasses the orchestrator
- [ ] Stage-1 training actually run: 30 epochs, lr=1e-4, warmup=200, augment, quit=fixed, freeze=0 (full unfreeze) on ~17,955 lines
- [ ] Download adapted model → `models/kraken/sophro_jer36_adapted.safetensors`

### Stage 2 — Sogdian delta fine-tune (NOT STARTED)
- [ ] `uv run msocr train-remote --manifest data/manifests/c2av-finetune.json --style-group c2av-syriac-finetune --base-model models/kraken/sophro_jer36_adapted.safetensors --output-model models/kraken/sophro_jer36_adapted_c2av.safetensors --epochs 200 --augment --warmup 200 --lr 5e-5 --quit early --min-epochs 20 --lag 10 --freeze-backbone 999999`
- [ ] Eval stage-2 model on c2av12 holdout, compare to baseline 71.43% CER / 28.57% accuracy (current best, `c2av_finetune_200ep`)

### Deferred
- Segmentation-axis fix (manual polygon QA). Bootstrap-polygons generated for c2av11 (5 lines) + c2av12 (19 lines) → `/tmp/msocr_polyqa/{c2av11,c2av12}_contact.png`. User deferred this to prioritize domain-adaptation. Auto-poly IoU ~0.85 vs manual, with outlier lines at 0.56. May still need this if stage 2 plateaus
- Pre-existing test failures in `tests/training/test_runpod_runner.py` (unrelated to ssh_exec fix, verified via git stash)

## Next-move recipe (for whoever picks this up)

1. **Quick path (manual drive of existing pod)**:
   ```bash
   ssh -i ~/.ssh/id_ed25519 -p 46304 root@174.94.157.109
   # apply patches (see _KRAKEN_CHECKPOINT_PATCH, _SAFETENSORS_SHARED_TENSOR_PATCH in orchestrator.py)
   # launch training in tmux:
   tmux new -d -s train 'cd /workspace && ketos train --base-model base.safetensors --manifest train_manifest.txt --validation val_manifest.txt --output /workspace/models/jer36-adapt/model --epochs 30 --warmup 200 --lr 1e-4 --augment --quit fixed --freeze 0 2>&1 | tee /workspace/train.log'
   # poll: tmux attach -t train, or tail /workspace/train.log
   # download when done: scp root@174.94.157.109:/workspace/models/jer36-adapt/model_*.safetensors models/kraken/sophro_jer36_adapted.safetensors
   ```
   Then kill pod `phpyu59p2flw8g` via RunPod API or CLI.
   **This bypasses the orchestrator bug entirely and gets training running in ~5 min.**

2. **Proper path (fix ssh_exec then relaunch)**: Add watchdog to `ssh_exec`:
   ```python
   # in poll loop, track last_recv_time
   if chan.recv_ready():
       last_recv_time = time.monotonic()
       ...drain...
   if time.monotonic() - last_recv_time > IDLE_TIMEOUT:  # e.g. 120s
       chan.close()
       raise RuntimeError("ssh_exec idle timeout — channel may be in zombie state")
   ```
   Or switch to `subprocess.run(["ssh", ...])` for setup commands (OS ssh handles EOF correctly).
   Then kill orchestrator + pod, relaunch. Costs ~10 min upload + ~$0.30 pod spin-up.

3. **After adapted model returns**: stage 2 (see command above), eval on c2av12, compare to 71.43% CER baseline.

## Useful environment/state references
- `.env`: `RunPod_API="<REDACTED: rotate if leaked>"` → export as `RUNPOD_API_KEY`
- Pod `phpyu59p2flw8g` (still RUNNING as of 2026-07-07 ~14:05). SSH: `ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 46304 root@174.94.157.109`
- Orchestrator log: `/tmp/jer36_adapt3.log` (268 lines, stuck at pip-install line)
- Polygon cache: `~/.cache/msocr/poly_cache/` (128 entries, pre-populated)
- Pre-converted base: `models/kraken/sophro_mhiro_syriac.safetensors`
- Stage-1 manifest: `data/manifests/ms-jer-36-adapt.json` (120/8/5 split)
- Stage-2 manifest: `data/manifests/c2av-finetune.json` (10/1/1)
- Stage-2 codec: `reports/c2av_union_codec.json` (37 base → 42 union)
- Stage-2 hyperparams (user-chosen): epochs=200, lr=5e-5, warmup=200, augment, quit=early, min-epochs=20, lag=10, freeze-backbone=999999 (effectively frozen)
- v5 plan: `docs/fixa-v5-updated-plan.md` (or `docs/fix-a-v5-plan.md`)
- v4 §1 falsified: freeze-rows proved problem is feature-level backbone mismatch, not classifier drift