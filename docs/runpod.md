# RunPod Remote Training Runbook

Runbook for fine-tuning a per-style-group Sogdian HTR model on a RunPod GPU Cloud Pod via `msocr train-remote`. Covers setup, invocation, manual recovery, and cost.

## 1. API key setup

1. Get a RunPod API key at <https://console.runpod.io/tokens>.
2. Export it in your shell (add to `~/.zshrc` / `~/.bashrc` to persist):

   ```bash
   export RUNPOD_API_KEY=XXXXXXXXXXXXXXXX
   ```

`msocr train-remote` refuses to run without `RUNPOD_API_KEY` set.

## 2. SSH key generation + upload

The runner SSH-es into the pod to exec `ketos train`. RunPod injects your public key at pod boot.

1. Generate a keypair (skip if you already have `~/.ssh/id_ed25519`):

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
   ```

2. Copy the public key:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. In the RunPod dashboard: **Settings → SSH Keys → Add SSH Key**, paste the `.pub` content, name it.

The default `--ssh-key ~/.ssh/id_ed25519` flag points at the private key locally; RunPod matches it to the uploaded public key.

## 3. Pod image build + push

`--pod-image` defaults to `msocr-kraken7:latest`. Build from `Dockerfile.train` and push to a registry RunPod can pull from (Docker Hub public is simplest; for private use RunPod's registry + **Settings → Registry Auth**).

```bash
# build (from repo root)
docker build -f Dockerfile.train -t msocr-kraken7:latest .

# tag for your registry, e.g. Docker Hub user "ajiap", then push
docker tag msocr-kraken7:latest ajiap/msocr-kraken7:latest
docker login && docker push ajiap/msocr-kraken7:latest
```

Point the CLI at the pushed image: `--pod-image ajiap/msocr-kraken7:latest`.

## 4. `msocr train-remote` invocation

Full example for one style-group. The runner submits the pod, SSH-execs `ketos train`, polls until the pod `EXITED`, scp-s the `.safetensors` artifact to `--output-model`, then terminates the pod. Evaluation runs locally after download.

```bash
RUNPOD_API_KEY=... uv run msocr train-remote \
  --manifest data/manifests/berlin-turfan-sogdian-v1.json \
  --style-group manichaean-early \
  --base-model models/kraken/openiti-arabic-base.safetensors \
  --output-model models/kraken/sogdian-manichaean-early.mlmodel \
  --reports-dir reports/ \
  --pod-gpu "RTX 4090" \
  --pod-image ajiap/msocr-kraken7:latest \
  --ssh-key ~/.ssh/id_ed25519 \
  --epochs 50 --min-epochs 20 --lag 10 --freeze-backbone 5000 --augment
```

Flags of note:
- `--pod-gpu` — RunPod GPU type id, default `RTX 4090`.
- `--ssh-key` — local private key path (default `~/.ssh/id_ed25519`).
- `--epochs` / `--min-epochs` / `--lag` — ketos early-stop knobs.
- `--freeze-backbone` — backbone params frozen during fine-tune.
- `--augment / --no-augment` — ketos augmentation toggle (default on).
- `--device` (default `cuda:0`), `--workers` (default `8`) — pod-side training args.

On success the fine-tuned model lands at `--output-model`; a CER/WER report lands under `--reports-dir`.

## 5. Manual recovery if download fails

By design the runner does **not** terminate the pod when the `scp` download fails — the artifact is still on the pod and you can grab it by hand.

1. Find the pod id and ip in the RunPod dashboard or the runner's last log line.
2. SSH in (RunPod pods run as `root`):

   ```bash
   ssh -i ~/.ssh/id_ed25519 root@<pod-ip>
   ls -lh /workspace/models/          # confirm the artifact is there
   ```

3. Pull it down (from your local machine):

   ```bash
   scp -i ~/.ssh/id_ed25519 \
     root@<pod-ip>:/workspace/models/manichaean-early.safetensors \
     models/kraken/manichaean-early.safetensors
   ```

4. Terminate the pod so you stop paying:

   ```bash
   runpod terminate-pod <pod-id>      # or dashboard → Pods → terminate
   ```

5. Re-run evaluation locally on the recovered model:

   ```bash
   uv run msocr evaluate \
     --manifest data/manifests/berlin-turfan-sogdian-v1.json \
     --style-group manichaean-early \
     --model models/kraken/manichaean-early.safetensors \
     --reports-dir reports/
   ```

If the pod died mid-training (host lost, GPU OOM), RunPod's persistent disk holds the latest `ketos` checkpoint under `/workspace/models/` — re-launch with `ketos train --resume`. See design §5.1.

## 6. Cost ballpark

- **RTX 4090** on RunPod Community: ~**$0.40/hr**.
- Fine-tune on ~100 pages × 50 epochs ≈ **1.5 hr** → **~$0.60** per style-group model.
- Add **~$0.10** for pod boot, image pull, idle overhead.
- Realistic per-style-group budget: **~$0.70**.

If a run fails after training (download fail), the pod is left running — terminate it (§5) or you keep paying. Multiple style-groups: one pod per group, run sequentially, budget `N × $0.70`.