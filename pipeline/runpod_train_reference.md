# RunPod Vienna Syriac Pretrain (Reference)

Source of truth:
- `pipeline/payne-smith_syriac_runpod_train.yaml`

This is a runnable reference for the default training path (RunPod + Ketos) with the Vienna Syriac GT.

**Short checklist (quick path)**
1. Install local tools: `runpodctl`, `rsync`, `git-lfs`, `ssh`.
2. Create RunPod Network Volume (100 GB) and GPU Pod (RTX A5000 recommended).
3. SSH into the pod, set up `uv` venv and install Python deps.
4. Download Vienna GT from Zenodo to `/workspace/dataset/vienna`.
5. Run Ketos pretrain on GPU; store model in `/workspace/models/pretrain/`.
6. Rsync model and logs back to local.

---

## Tools to install (local)
- `runpodctl`
- `rsync`
- `git` + `git-lfs`
- `ssh` (OpenSSH client)
- A local SSH key pair (recommended: `~/.ssh/id_ed25519`)

---

## Zenodo dataset download (Vienna Syriac GT)
These are the exact URLs used in `pipeline/archieve/colab_t4_train_vienna.ipynb`:
- `https://zenodo.org/records/14714089/files/images.zip?download=1`
- `https://zenodo.org/records/14714089/files/page.zip?download=1`

Example (RunPod pod, on `/workspace`):
```bash
mkdir -p /workspace/dataset/vienna
cd /workspace/dataset/vienna
curl -L -o images.zip "https://zenodo.org/records/14714089/files/images.zip?download=1"
curl -L -o page.zip   "https://zenodo.org/records/14714089/files/page.zip?download=1"
unzip -q images.zip
unzip -q page.zip
```

---

## RunPod workflow (expanded)

### 1) Create a Network Volume (persistent)
- Name: `syriac-ocr-vol`
- Size: 100 GB
- Region: closest to your GPU availability (YAML defaults to `EUR-IS-1`)
- Mount path: `/workspace`

### 2) Create a GPU Pod
Recommended:
- GPU: RTX A5000
- Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- Compute type: `ON_DEMAND`
- Attach network volume to `/workspace`
- Enable SSH

### 3) Pod setup (run once)
```bash
# 1) Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 2) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3) Create venv and install deps
cd /workspace
uv venv --python 3.10
source .venv/bin/activate
uv pip install kraken numpy pillow tqdm pandas scikit-image scikit-learn pyyaml orjson rich

# 4) Create directories
mkdir -p /workspace/dataset/{vienna,train,validation,holdout}
mkdir -p /workspace/models/{pretrain,finetune}
mkdir -p /workspace/logs
```

### 4) Dataset preparation (Vienna)
```bash
# Validate Vienna GT
python - <<'EOF'
import glob
from pathlib import Path
xml_files = glob.glob("/workspace/dataset/vienna/**/*.xml", recursive=True)
print(f"Total XML files: {len(xml_files)}")
missing = []
for xml in xml_files:
    p = Path(xml)
    for ext in [".png", ".jpg", ".tif"]:
        if p.with_suffix(ext).exists():
            break
    else:
        missing.append(xml)
print("Missing images:", len(missing))
EOF

# Split dataset (train/validation/holdout)
python - <<'EOF'
import glob, random, shutil, json
from pathlib import Path
SEED = 42
random.seed(SEED)
xml_files = sorted(glob.glob("/workspace/dataset/vienna/**/*.xml", recursive=True))
random.shuffle(xml_files)

n = len(xml_files)
n_holdout = max(1, int(n * 0.05))
n_validation = max(1, int(n * 0.10))

holdout = xml_files[:n_holdout]
validation = xml_files[n_holdout:n_holdout + n_validation]
train = xml_files[n_holdout + n_validation:]

def copy_split(files, dest_dir):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for xml in files:
        p = Path(xml)
        shutil.copy2(xml, dest / p.name)
        for ext in [".png", ".jpg", ".tif"]:
            img = p.with_suffix(ext)
            if img.exists():
                shutil.copy2(img, dest / img.name)
                break

copy_split(train, "/workspace/dataset/train")
copy_split(validation, "/workspace/dataset/validation")
copy_split(holdout, "/workspace/dataset/holdout")

manifest = {
  "seed": SEED,
  "total": n,
  "train": [str(Path(f).name) for f in train],
  "validation": [str(Path(f).name) for f in validation],
  "holdout": [str(Path(f).name) for f in holdout],
}
with open("/workspace/dataset/split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Manifest saved to /workspace/dataset/split_manifest.json")
EOF
```

### 5) Vienna pretrain (Ketos)
```bash
source /workspace/.venv/bin/activate

# Recommended: run inside tmux
# tmux new -s vienna_pretrain

ketos train \
  -f xml \
  --base-dir R \
  --augment \
  --lag 10 \
  --min-epochs 30 \
  --device cuda:0 \
  --output /workspace/models/pretrain/vienna_serto \
  /workspace/dataset/train/*.xml \
  2>&1 | tee /workspace/logs/vienna_pretrain_$(date +%Y%m%d_%H%M%S).log
```

### 6) Retrieve results (RunPod → local)
```bash
rsync -avzP -e "ssh -p TCP_PORT_22 -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/models/ ./models/

rsync -avzP -e "ssh -p TCP_PORT_22 -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/logs/ ./logs/
```

---

## Notes
- Training is **Ketos only** (Kraken training docs now center on `ketos train`).
- RunPod persistent pods are required; serverless is not suitable for multi-hour training.
- Vienna GT acquisition defaults to **Zenodo zip** (Git LFS is optional fallback).
- This project treats RunPod as the **default training method**.
