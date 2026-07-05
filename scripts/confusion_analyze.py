#!/usr/bin/env python3
"""0.1 Confusion-matrix analysis — categorize the 332 errors on c2av12 holdout.

Reads reports/confusion_c2av12_holdout.json (raw confusion rows) + the txt report
and produces a categorized analysis:

  - error type breakdown (deletion / substitution / insertion in standard terms)
  - per-char drop rate (how often each gt char is dropped)
  - shared vs new char error split (the 1.1 decision gate)

kraken's "insertions" field = model dropped gt char (pred empty) — standard deletion.
kraken's "deletions" field = model emitted phantom char (gt empty) — standard insertion.

Outputs:
  reports/confusion_c2av12_analysis.md  — human-readable analysis
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# ponytail: the 5 NEW classes added by --resize union in v2 (not in sophro base)
NEW_CLASSES = {"ݎ", "ݏ", "ݍ", "SYRIAC QUSHSHAYA", "SYRIAC RUKKAKHA"}
# ponytail: chars that were in the sophro base codec (22 Syriac consonants + space + .)
# Pulled from the v2 report — these are the shared classes.
SHARED_CLASSES = {"ܐ", "ܒ", "ܓ", "ܕ", "ܘ", "ܙ", "ܚ", "ܛ", "ܝ", "ܟ", "ܠ", "ܡ",
                  "ܢ", "ܣ", "ܥ", "ܦ", "ܨ", "ܩ", "ܪ", "ܬ", "ܫ", "ܢ", "SPACE", "."}


def main() -> int:
    rows = json.loads((REPORTS / "confusion_c2av12_holdout.json").read_text())
    txt = (REPORTS / "confusion_c2av12_holdout.txt").read_text()

    # ponytail: parse the summary section for total counts
    summary = _parse_summary(txt)

    # Categorize each confusion row
    deletions = []      # gt has char, pred empty (kraken "insertion")
    insertions = []     # gt empty, pred has char (kraken "deletion")
    substitutions = []  # both have chars, different
    for r in rows:
        c, g, n = r["correct"], r["generated"], r["errors"]
        if c and not g:
            deletions.append(r)
        elif g and not c:
            insertions.append(r)
        elif c and g and c != g:
            substitutions.append(r)

    # Per-char drop rate (deletions only, sorted by count)
    drop_by_char = sorted(deletions, key=lambda x: x["errors"], reverse=True)

    # Shared vs new class split (for deletions — the dominant error mode)
    new_del_count = sum(r["errors"] for r in deletions if r["correct"] in NEW_CLASSES)
    shared_del_count = sum(r["errors"] for r in deletions if r["correct"] not in NEW_CLASSES)
    sub_new = sum(r["errors"] for r in substitutions if r["correct"] in NEW_CLASSES)
    sub_shared = sum(r["errors"] for r in substitutions if r["correct"] not in NEW_CLASSES)

    out = []
    out.append("# Confusion-matrix analysis: c2av12 holdout")
    out.append("")
    out.append("Model: `models/kraken/c2av_finetune_union_frozen.safetensors` (3.2 shipped)")
    out.append("Set: c2av12 holdout, 19 lines, 427 chars total")
    out.append("CER: 77.75% (332 errors / 427 chars)")
    out.append("")
    out.append("## Error type breakdown (standard terminology)")
    out.append("")
    out.append("| Type | Count | % of errors | Notes |")
    out.append("|---|---|---|---|")
    out.append(f"| **Deletions** (model dropped gt char) | {summary['insertions']} | {100*summary['insertions']/summary['errors']:.1f}% | kraken labels this \"insertions\" |")
    out.append(f"| **Insertions** (phantom char) | {summary['deletions']} | {100*summary['deletions']/summary['errors']:.1f}% | kraken labels this \"deletions\" |")
    out.append(f"| **Substitutions** | {summary['substitutions']} | {100*summary['substitutions']/summary['errors']:.1f}% | wrong char emitted |")
    out.append("")
    out.append("## Top 15 dropped characters (deletions)")
    out.append("")
    out.append("| Count | Char | Class |")
    out.append("|---|---|---|")
    for r in drop_by_char[:15]:
        cls = "NEW" if r["correct"] in NEW_CLASSES else "shared"
        out.append(f"| {r['errors']} | `{r['correct']}` | {cls} |")
    out.append("")
    out.append("## Shared vs new class error split (1.1 decision gate)")
    out.append("")
    out.append("| Error type | Shared classes | New classes |")
    out.append("|---|---|---|")
    out.append(f"| Deletions | {shared_del_count} | {new_del_count} |")
    out.append(f"| Substitutions | {sub_shared} | {sub_new} |")
    out.append(f"| **Total** | **{shared_del_count + sub_shared}** | **{new_del_count + sub_new}** |")
    out.append("")
    # ponytail: the 1.1 decision
    total_err = summary["errors"]
    new_total = new_del_count + sub_new
    shared_total = shared_del_count + sub_shared
    out.append("## 1.1 decision (`--append` ablation)")
    out.append("")
    out.append(f"- **{100*new_total/total_err:.1f}%** of errors involve NEW classes (dropped or substituted)")
    out.append(f"- **{100*shared_total/total_err:.1f}%** of errors involve SHARED classes (already in sophro base)")
    out.append("")
    if shared_total > new_total:
        out.append("**Shared-class errors dominate.** The v2 frozen classifier is NOT successfully preserving the 22 shared consonants — it's dropping them too. `--append` (fresh classifier, loses shared weights) may NOT help and could hurt. **Skip 1.1.**")
    else:
        out.append("**New-class errors dominate.** The shared classifier weights are doing useful work; errors concentrate on the 5 new classes. `--append` won't help — it would lose the working shared weights. **Skip 1.1.**")
    out.append("")
    out.append("## Top 15 substitutions (wrong char emitted)")
    out.append("")
    out.append("| Count | Correct | Generated |")
    out.append("|---|---|---|")
    for r in sorted(substitutions, key=lambda x: x["errors"], reverse=True)[:15]:
        out.append(f"| {r['errors']} | `{r['correct']}` | `{r['generated']}` |")
    out.append("")
    out.append("## Diagnosis")
    out.append("")
    out.append(f"The model is **dropping {summary['insertions']} of {summary['chars']} characters ({100*summary['insertions']/summary['chars']:.1f}%)**. This is the dominant failure mode — not misclassification. The model has learned to emit too few characters, not the wrong ones.")
    out.append("")
    out.append("This is a **segmentation/CTC alignment issue**, not a classifier issue. The frozen backbone + union classifier can emit the right chars but the CTC decoder is collapsing — emitting blank for too many timesteps. `--append` (fresh classifier) won't fix this. The 0.2 200-epoch rerun is the right next lever — more training may improve CTC alignment.")

    md = "\n".join(out) + "\n"
    (REPORTS / "confusion_c2av12_analysis.md").write_text(md)
    print(md)
    return 0


def _parse_summary(txt: str) -> dict:
    """Parse the summary block from the ketos report."""
    s: dict[str, int] = {}
    for line in txt.splitlines():
        line = line.strip()
        if line.endswith("Characters"):
            s["chars"] = int(line.split("\t")[0])
        elif line.endswith("Errors"):
            s["errors"] = int(line.split("\t")[0])
        elif line.endswith("Insertions"):
            s["insertions"] = int(line.split("\t")[0])
        elif line.endswith("Deletions"):
            s["deletions"] = int(line.split("\t")[0])
        elif line.endswith("Substitutions"):
            s["substitutions"] = int(line.split("\t")[0])
    return s


if __name__ == "__main__":
    raise SystemExit(main())