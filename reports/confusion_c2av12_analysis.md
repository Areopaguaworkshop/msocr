# Confusion-matrix analysis: c2av12 holdout

Model: `models/kraken/c2av_finetune_union_frozen.safetensors` (3.2 shipped)
Set: c2av12 holdout, 19 lines, 427 chars total
CER: 77.75% (332 errors / 427 chars)

## Error type breakdown (standard terminology)

| Type | Count | % of errors | Notes |
|---|---|---|---|
| **Deletions** (model dropped gt char) | 277 | 83.4% | kraken labels this "insertions" |
| **Insertions** (phantom char) | 4 | 1.2% | kraken labels this "deletions" |
| **Substitutions** | 51 | 15.4% | wrong char emitted |

## Top 15 dropped characters (deletions)

| Count | Char | Class |
|---|---|---|
| 39 | `ܝ` | shared |
| 27 | `ܘ` | shared |
| 26 | `SPACE` | shared |
| 24 | `ܐ` | shared |
| 19 | `ܬ` | shared |
| 18 | `ܪ` | shared |
| 18 | `ܢ` | shared |
| 17 | `SYRIAC QUSHSHAYA` | NEW |
| 11 | `ݎ` | NEW |
| 9 | `ܩ` | shared |
| 9 | `ܓ` | shared |
| 9 | `ܙ` | shared |
| 8 | `ܫ` | shared |
| 8 | `ܡ` | shared |
| 7 | `.` | shared |

## Shared vs new class error split (1.1 decision gate)

| Error type | Shared classes | New classes |
|---|---|---|
| Deletions | 243 | 34 |
| Substitutions | 47 | 4 |
| **Total** | **290** | **38** |

## 1.1 decision (`--append` ablation)

- **11.4%** of errors involve NEW classes (dropped or substituted)
- **87.3%** of errors involve SHARED classes (already in sophro base)

**Shared-class errors dominate.** The v2 frozen classifier is NOT successfully preserving the 22 shared consonants — it's dropping them too. `--append` (fresh classifier, loses shared weights) may NOT help and could hurt. **Skip 1.1.**

## Top 15 substitutions (wrong char emitted)

| Count | Correct | Generated |
|---|---|---|
| 3 | `ܝ` | `SPACE` |
| 3 | `ܝ` | `ܬ` |
| 3 | `ܫ` | `ܐ` |
| 2 | `ܘ` | `ܪ` |
| 2 | `SPACE` | `SYRIAC QUSHSHAYA` |
| 2 | `ܢ` | `ܬ` |
| 2 | `ܐ` | `ܢ` |
| 2 | `ܐ` | `SYRIAC QUSHSHAYA` |
| 2 | `ܘ` | `SPACE` |
| 2 | `ܙ` | `ܬ` |
| 1 | `ܨ` | `ܐ` |
| 1 | `ܝ` | `ܣ` |
| 1 | `ܨ` | `ܬ` |
| 1 | `ܝ` | `ܫ` |
| 1 | `ݍ` | `ܝ` |

## Diagnosis

The model is **dropping 277 of 427 characters (64.9%)**. This is the dominant failure mode — not misclassification. The model has learned to emit too few characters, not the wrong ones.

This is a **segmentation/CTC alignment issue**, not a classifier issue. The frozen backbone + union classifier can emit the right chars but the CTC decoder is collapsing — emitting blank for too many timesteps. `--append` (fresh classifier) won't fix this. The 0.2 200-epoch rerun is the right next lever — more training may improve CTC alignment.
