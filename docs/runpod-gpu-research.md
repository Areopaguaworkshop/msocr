# Kraken HTR Training on RunPod — GPU Selection & Cost Analysis

> Date: June 2026
> Scope: Training a Kraken recognition model from scratch for Sogdian manuscripts (3,000–10,000 line-image/transcription pairs) on RunPod cloud GPUs.
> Source: deep web research via @librarian, mid-2026.

## TL;DR

| Item | Recommendation |
|------|---------------|
| **GPU** | **RTX 4090, Community Cloud Spot, $0.34/hr** |
| **Budget fallback** | RTX 3090, Community Cloud Spot, $0.22/hr |
| **5,000 lines cost** | **~$1–$3** (spot), ~$2–$6 (secure) |
| **10,000 lines cost** | **~$2–$5** (spot), ~$4–$11 (secure) |
| **Multi-GPU?** | No — Kraken doesn't support it. Single GPU only. |
| **Precision** | `--precision bf16-mixed` (requires Ampere+) |
| **Dataset format** | Pre-compiled binary Arrow (`.arrow`) |
| **Storage** | RunPod Network Volume, $0.07/GB/month |
| **Checkpointing** | Kraken 7.0 `.ckpt` files — full state, spot-safe |

The entire project's GPU compute budget is likely under $20 total, even with multiple training attempts.

---

## 1. Kraken GPU Requirements

### Model Size & Architecture

Kraken's default recognition model is a VGSL-specified CRNN+CTC network. From the [training tutorial](https://kraken.re/6.0.0/tutorials/training.html):

> **Trainable params: 4.1 M**
> **Total estimated model params size (MB): 16**

Default spec: `[1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]`

Recommended "large/complicated" manuscript spec (use this for Sogdian): `[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]` — still single-digit millions of params.

**Bottom line**: tiny model by modern standards. VRAM is not the bottleneck. Even 16GB is vastly more than needed for weights + activations + optimizer states.

### CUDA, Precision, Tensor Cores

- **CUDA required** for GPU acceleration (`--device cuda:0`). Kraken uses PyTorch Lightning.
- **bf16-mixed precision** (`--precision bf16-mixed`) explicitly recommended: *"On GPU, bfloat16 can lead to considerable speedup"* and *"significant speedup without any loss in accuracy."*
- **Tensor Cores utilized** — confirmed in GitHub issue #571 (RTX 3090 triggers PyTorch's tensor-core warning). All GPUs from RTX 20-series onward have tensor cores. Ampere (3090, A-series) = 3rd-gen; Ada Lovelace (4090, L40S) = 4th-gen; Hopper (H100) = 4th-gen + FP8.
- **bf16 requires Ampere+** (RTX 3090+, A100+, 4090+). Older GPUs (Turing, Volta) only support fp16.

### Single-GPU Only — No Multi-GPU in ketos train

**Critical finding.** Kraken's `ketos train` CLI exposes a single `--device` flag. No multi-GPU support in the ketos training path:

- `KrakenTrainer` wraps PyTorch Lightning's `Trainer` (which *could* support DDP), but ketos CLI only passes a single device index.
- No `DistributedDataParallel`, `DataParallel`, or multi-GPU references in docs, issues, or source for recognition training.
- `--device` accepts only one value: `cpu`, `cuda:0`, `cuda:1`, etc.

**Implication**: single-GPU only. Multi-GPU instances waste the extra GPUs. Simplifies selection — pick the fastest single GPU you can afford.

### CPU / Data Loading

- **Binary dataset format (Arrow) essential**: *"Binary datasets drastically improve loading performance allowing the saturation of most GPUs with minimal computational overhead."* Pre-compile with `ketos compile -f xml --workers 8`.
- **`--workers`** for data loading (recommend 4–8, especially with `--augment`).
- **`--threads`** for intra-op parallelization (OpenMP, BLAS).
- **Memory-mapped loading**: binary datasets can be larger than RAM. Place on local storage (not NFS).
- **CPU bottleneck risk**: with augmentation + small GPU, data loading can bottleneck. 4090 is fast enough that 4–8 workers may be needed to keep it fed. RunPod pods typically provide 8–16 vCPUs.

### CPU-bound or GPU-bound?

**GPU-bound for training.** CRNN forward/backward passes dominate runtime. Binary Arrow + multiple workers is efficient enough that even a 4090 should stay saturated. Docs: *"Use option -B to scale batch size until GPU utilization reaches 100%."*

### Training Time Guidance (from docs)

> *"Training a network will take some time on a modern computer, even with the default parameters. While the exact time required is unpredictable as training is somewhat random, a rough guide is that accuracy seldom improves after 50 epochs reached between 8 and 24 hours of training on a normal desktop PC. Training on a GPU can significantly speed up the process, making training runs of under an hour possible."*

The "under an hour" claim is for ~788 lines. For 5,000–10,000 lines with augmentation and the larger manuscript architecture, expect proportionally longer.

---

## 2. RunPod GPU Pricing Table (Mid-2026)

Prices from [RunPod pricing](https://www.runpod.io/pricing), [GPU models](https://www.runpod.io/gpu-models), DeployBase (March 2026), GPUPerHour, ByteCosts (June 2026), cloudgpuprice.com, costbench.com. **Prices fluctuate; verify live before deploying.**

| GPU | VRAM | Architecture | Community Cloud (Spot) | Secure Cloud (On-Demand) | Notes |
|-----|------|-------------|----------------------|------------------------|-------|
| **RTX A4000** | 16GB | Ampere | ~$0.12/hr | ~$0.20/hr | Cheapest; 16GB sufficient; slower compute |
| **RTX A5000** | 24GB | Ampere | $0.16/hr | $0.27/hr | Budget sweet spot; 24GB, decent compute |
| **RTX 3090** | 24GB | Ampere | **$0.22/hr** | $0.39–$0.46/hr | Budget champion; widely available; 3rd-gen tensor |
| **RTX 4090** | 24GB | Ada Lovelace | **$0.34/hr** | $0.60–$0.69/hr | **Price/perf sweet spot**; 4th-gen tensor; ~2.3× faster than 3090 |
| **RTX 5090** | 32GB | Blackwell | $0.69/hr | $0.99–$1.22/hr | Newest consumer; limited availability; overkill |
| **L4** | 24GB | Ada Lovelace | $0.44/hr | $0.78/hr | Data center; lower TDP; slower than 4090 |
| **L40** | 48GB | Ada Lovelace | $0.69/hr | $0.99/hr | 48GB; overkill |
| **L40S** | 48GB | Ada Lovelace | $0.79/hr | $0.86–$1.40/hr | Higher-clocked L40; overkill |
| **RTX A6000** | 48GB | Ampere | $0.33/hr | $0.49/hr | 48GB at good price; slower compute than 4090 |
| **A40** | 48GB | Ampere | $0.35/hr | $0.44/hr | Data center Ampere; similar to A6000 |
| **A100 PCIe 40GB** | 40GB | Ampere | ~$0.79/hr | ~$1.19/hr | Data center; overkill |
| **A100 PCIe 80GB** | 80GB | Ampere | $1.19/hr | $1.39/hr | Overkill; 3–5× cost of 4090 for marginal benefit |
| **A100 SXM 80GB** | 80GB | Ampere | $1.39/hr | $1.49/hr | Higher bandwidth; overkill |
| **H100 PCIe 80GB** | 80GB | Hopper | $1.99/hr | $2.89/hr | Massive overkill for 4.1M param model |
| **H100 SXM 80GB** | 80GB | Hopper | $2.69/hr | $3.29/hr | Even more overkill |

**Pricing notes**:
- Community Cloud = spot/preemptible (~5 min interruption notice). ~40–60% cheaper than Secure.
- Secure Cloud = guaranteed uptime, Tier 3/4 data centers, SOC 2. ~1.5–2× Community.
- Per-second billing on both tiers. Stop pod → billing stops.
- Spot prices fluctuate ±15% on supply/demand.
- RTX 3090 and 4090 are the most consistently available Community Cloud GPUs.

---

## 3. Best Price/Performance for Kraken Training

### Analysis

Kraken's model is ~4.1M params. **Compute-bound**, not memory-bound. Key metric: **FP32/FP16 tensor FLOPs per dollar**.

| GPU | Approx. FP32 TFLOPS | Spot $/hr | TFLOPS/$ | Relative value |
|-----|-------------------|-----------|----------|---------------|
| RTX A5000 | ~27.8 | $0.16 | 174 | 1.00× (baseline) |
| RTX 3090 | ~35.6 | $0.22 | 162 | 0.93× |
| RTX 4090 | ~82.6 | $0.34 | **243** | **1.40×** |
| RTX 5090 | ~100+ (est.) | $0.69 | ~145 | 0.83× |
| A100 PCIe | ~19.5 (FP32) / 312 (FP16 tensor) | $1.19 | ~262 (FP16) | 1.51× (FP16 only) |
| H100 PCIe | ~60 (FP32) / 990 (FP16 tensor) | $1.99 | ~497 (FP16) | 2.86× (FP16 only) |

**Caveat**: A100/H100 FP16 tensor numbers look great on paper, but Kraken's tiny model won't saturate those tensor cores. The model is too small to benefit from the massive parallelism of data center GPUs. You'd pay 3–6× more per hour for compute you can't fully utilize.

### Recommendation: RTX 4090 (Community Cloud Spot)

**The RTX 4090 at $0.34/hr spot is the unambiguous sweet spot.**

Rationale:
1. **~2.3× faster than RTX 3090** in raw compute for only **1.55× the price** — better TFLOPS/$.
2. **4th-gen Tensor Cores** with bf16 support — Kraken's recommended `bf16-mixed` precision runs natively.
3. **24GB VRAM** is far more than Kraken needs, allowing large batch sizes (32–64+) to fully saturate the GPU.
4. **Widely available** on Community Cloud — the most common GPU after the 3090.
5. **Total cost is trivially low** — even a 10-hour training run costs ~$3.40 on spot.

**Budget alternative**: RTX 3090 at $0.22/hr spot. If cost-constrained and don't mind ~2× longer training, cheapest 24GB option with bf16. Total cost for 10k-line run still under $5.

**When to consider Secure Cloud**: if you cannot tolerate interruption risk (tight deadline, no checkpointing infra), RTX 4090 Secure at $0.69/hr is still reasonable. But with Kraken 7.0+'s Lightning checkpointing (`.ckpt` files with full optimizer/scheduler state), spot interruptions are trivially recoverable — restart from last checkpoint.

**What NOT to use**: A100, H100, L40S, RTX 5090. All 2–6× more expensive per hour with no proportional benefit for a 4.1M param CRNN. The model is simply too small to utilize their massive compute capacity.

---

## 4. Cost Estimates

### Methodology

Kraken-specific GPU benchmarks are **scarce** — no published "lines per second" numbers. Estimates extrapolate from:

1. **Official doc guidance**: 8–24 hours CPU → "under an hour" GPU for ~788 lines. Implies ~10–20× GPU speedup.
2. **Model size**: 4.1M params, ~16MB. Each training step is a small CRNN forward/backward pass.
3. **GPU relative speeds**: RTX 4090 ≈ 2.3× RTX 3090 for FP32 compute.
4. **Batch size scaling**: default batch=1. With 24GB VRAM, batch=16–32 is easily achievable, reducing steps/epoch proportionally.
5. **Augmentation overhead**: recommended manuscript spec uses `--augment` (CPU-side). With 4–8 workers, should not bottleneck a 4090.

### Assumptions

- Binary Arrow dataset format (pre-compiled)
- `--precision bf16-mixed`
- `--augment` enabled with `--workers 4–8`
- Manuscript architecture (larger VGSL spec)
- Batch size tuned to saturate GPU (8–32)
- 40 epochs (midpoint of 30–50; early stopping may terminate earlier)
- ~10% validation split

### Estimates

| Scenario | GPU | Est. Time/Epoch | Total Time (40 epochs) | Spot Cost | Secure Cost |
|----------|-----|----------------|----------------------|-----------|-------------|
| **5,000 lines** | RTX 3090 | ~6–10 min | **4–7 hours** | **$0.88–$1.54** | $1.56–$3.22 |
| **5,000 lines** | RTX 4090 | ~3–5 min | **2–3.5 hours** | **$0.68–$1.19** | $1.20–$2.42 |
| **10,000 lines** | RTX 3090 | ~12–20 min | **8–13 hours** | **$1.76–$2.86** | $3.12–$5.98 |
| **10,000 lines** | RTX 4090 | ~6–10 min | **4–7 hours** | **$1.36–$2.38** | $2.40–$4.83 |

**Conservative upper bound** (slow convergence, 50 epochs, batch=1, heavy augmentation):

| Scenario | GPU | Est. Total Time | Spot Cost | Secure Cost |
|----------|-----|----------------|-----------|-------------|
| 5,000 lines, 50 epochs | RTX 4090 | ~6–8 hours | **$2.04–$2.72** | $4.14–$5.52 |
| 10,000 lines, 50 epochs | RTX 4090 | ~12–16 hours | **$4.08–$5.44** | $8.28–$11.04 |

### The Math (RTX 4090, 10,000 lines, 40 epochs, spot)

```
10,000 lines ÷ batch_size 16 = 625 steps/epoch
~0.5 sec/step (CRNN forward+backward, bf16, 4090) = ~5.2 min/epoch
40 epochs × 5.2 min = ~3.5 hours
3.5 hrs × $0.34/hr = $1.19
Add ~20% overhead (validation, checkpoint I/O, data loading stalls): ~$1.43
```

### Realistic Total Cost Range

**For a complete from-scratch training run on RTX 4090 Community Cloud spot:**

| Dataset Size | Expected Cost (Spot) | Expected Cost (Secure) |
|-------------|---------------------|----------------------|
| 5,000 lines | **$1–$3** | $2–$6 |
| 10,000 lines | **$2–$5** | $4–$11 |

These are **trivially low costs**. Even the pessimistic upper bound for 10,000 lines on Secure Cloud is ~$11. The entire project's GPU compute budget is likely under $20 total, even with multiple training attempts.

**Flagged uncertainty**: extrapolations. Kraken-specific GPU benchmarks don't exist in published form. The doc's "under an hour" claim is for ~788 lines on an unspecified GPU. Actual times depend on line image dimensions (Sogdian manuscript lines may be wider than average), augmentation intensity, and convergence speed. Budget 2× the optimistic estimate for safety.

---

## 5. RunPod Practical Considerations for Kraken

### Template Selection

RunPod offers pre-built **PyTorch** templates with CUDA and Python pre-installed. Choose **PyTorch 2.x + CUDA 12.x** (latest available):
- PyTorch with CUDA out of the box
- Python 3.12 (or installable via conda/uv)
- NVIDIA drivers pre-configured

Alternatively, bring your own Docker image. The project's `Dockerfile.train` can be built and pushed to Docker Hub or RunPod's container registry.

### Persistent Storage & Dataset Mounting

**Network Volume** is the right choice:
- $0.07/GB/month (<1TB) or $0.05/GB/month (>1TB)
- Persists across pod restarts and terminations
- Mount to `/workspace` or custom path
- Upload pre-compiled `.arrow` dataset + existing model checkpoints here

**Workflow**:
1. Create a Network Volume (e.g., 50GB — ~$3.50/month)
2. Upload dataset: `scp` or `rsync` your `.arrow` file + XML/images to the volume
3. Deploy a pod, attach the volume
4. Run training; checkpoints write to the volume
5. If spot-interrupted: deploy a new pod, attach same volume, resume from `.ckpt`

**Important**: Kraken docs say *"put them in a place where they can be memory mapped during training (local storage, not NFS or similar)."* RunPod Network Volumes are block storage attached as local disks — they support memory mapping. Fine.

### Checkpointing & Spot Interruption Recovery

Kraken 7.0+ (released April 2026) produces **PyTorch Lightning `.ckpt` checkpoints** with full training state:
> *"Checkpoint files include full training state (model weights, optimizer state, scheduler state, epoch/step counters, and serialized training config), enabling exact continuation of interrupted runs."*

**Ideal for spot instances**. If preempted:
1. The `.ckpt` file on the Network Volume survives
2. Deploy a new pod, attach the same volume
3. Resume with: `ketos -d cuda:0 train -f binary --load last.ckpt dataset.arrow`

Kraken also auto-converts the best checkpoint to `.safetensors` or `.mlmodel` at training end.

### Gotchas

1. **CUDA/cuDNN version compatibility**: Kraken pins PyTorch versions. The RunPod PyTorch template may have a different PyTorch/CUDA combo. Test locally first, or use your own Docker image with pinned dependencies.
2. **Community Cloud hardware variance**: different hosts have different CPU/RAM/disk speeds. For Kraken this matters less (GPU-bound), but very slow disk I/O could affect binary dataset loading. If I/O bottlenecks, switch to Secure.
3. **No multi-GPU**: don't pay for multi-GPU instances. Kraken can't use them.
4. **RTL direction**: Sogdian is RTL. Ensure dataset XML uses `horizontal-rl` reading direction. Kraken supports this natively.
5. **First epoch is slow**: Kraken preprocesses and encodes the dataset on first epoch. Subsequent epochs are faster. Don't panic if epoch 1 takes 2× longer.
6. **Storage costs add up**: a 50GB Network Volume costs ~$3.50/month. If kept for a month between training runs, that's more than the compute cost. Delete or downsize when done.

---

## 6. Alternatives (Brief)

| Platform | Cheapest Comparable GPU | Notes |
|----------|------------------------|-------|
| **Vast.ai** | RTX 3090 from ~$0.02/hr, RTX 4090 from ~$0.13/hr | Even cheaper than RunPod; less reliable; variable host quality; good for interruptible training |
| **Lambda Labs** | A100 80GB at $1.48/hr on-demand | More expensive; consistent data center hardware; no spot tier; good if you need guaranteed A100 |
| **Modal** | Serverless GPU, ~$0.50–$2.00/hr equivalent | Good for bursty jobs; 15-min timeout problematic for long training; not ideal for 5k+ line training |
| **HuggingFace ZeroGPU** | Free (A100/T4) | Limited to small workloads; not suitable for multi-hour training runs |
| **Paperspace** | A4000 from ~$0.40/hr | Similar to RunPod Secure pricing; less GPU variety |

**Verdict**: RunPod Community Cloud is the right choice. Vast.ai is cheaper but less reliable. For a $2–$5 total training cost, the reliability difference doesn't justify switching.

---

## Sources

- Kraken official docs: https://kraken.re/6.0.0/training/rectrain.html, https://kraken.re/6.0.0/tutorials/training.html
- Kraken GitHub: https://github.com/mittagessen/kraken (issues #571, #711; source at `kraken/ketos.py`, `kraken/lib/train.py`)
- Kraken 7.0 release notes: https://github.com/mittagessen/kraken/releases/tag/7.0
- RunPod pricing: https://www.runpod.io/pricing, https://www.runpod.io/gpu-models
- DeployBase pricing guide (March 2026): https://deploybase.ai/articles/runpod-gpu-pricing
- GPUPerHour RunPod tracker: https://gpuperhour.com/providers/runpod
- ByteCosts RunPod pricing (June 2026): https://bytecosts.com/gpu/runpod/
- cloudgpuprice.com RunPod review (June 2026): https://cloudgpuprice.com/provider/runpod/
- costbench.com RunPod pricing: https://costbench.com/software/ai-gpu-cloud/runpod/
- RunPod GPU model pages: https://www.runpod.io/gpu-models/rtx-3090, https://www.runpod.io/gpu-models/rtx-4090
- Digital Orientalist Kraken tutorial: https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/