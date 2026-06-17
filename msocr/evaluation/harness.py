"""Thin evaluation harness wrapping `ketos test`.

Per the design doc (D6): wrap ketos test, aggregate per-manuscript and
per-style-group, write JSON + Markdown. No invented metrics.

Ponytail: this is rung 4 — reuse what ketos test already reports
(CER, WER, case-insensitive CER, per-character accuracy, char-confusion).
We only parse its stdout and aggregate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from msocr.data.manifest import load_frozen_manifest, iter_style_group_cases
from msocr.training.ketos_trainer import KetosTrainer


_CER_RE = re.compile(r"CER:\s*([\d.]+)", re.IGNORECASE)
_WER_RE = re.compile(r"WER:\s*([\d.]+)", re.IGNORECASE)
_ACC_RE = re.compile(r"Accuracy:\s*([\d.]+)", re.IGNORECASE)


def _parse_ketos_stdout(stdout: str) -> dict[str, float]:
    """Extract CER/WER/Accuracy from ketos test stdout. Returns {} for missing."""
    out = {}
    if m := _CER_RE.search(stdout):
        out["cer"] = float(m.group(1))
    if m := _WER_RE.search(stdout):
        out["wer"] = float(m.group(1))
    if m := _ACC_RE.search(stdout):
        out["accuracy"] = float(m.group(1))
    return out


def run_evaluation(
    manifest_path: str,
    style_group_id: str,
    model_path: str,
    reports_dir: str,
    config: dict | None = None,
) -> dict[str, Any]:
    """Run ketos test on each manuscript in the style_group's holdout partition,
    aggregate per-manuscript and per-style-group, write JSON + Markdown report."""
    manifest = load_frozen_manifest(manifest_path)
    cases = list(iter_style_group_cases(manifest, style_group_id, partition="holdout"))

    # ponytail: a per-manuscript config is just the global config with this manifest's
    # dataset section. Could be fancier but YAGNI until we have >1 style_group.
    trainer = KetosTrainer(config or {
        "dataset": {"format_type": "xml"},
        "model": {"spec": "placeholder"},
        "training": {"epochs": 0, "device": "cpu", "workers": 1},
        "output": {"model_prefix": str(Path(model_path).with_suffix(""))},
    })

    per_manuscript: dict[str, dict[str, float]] = {}
    for case in cases:
        stdout = trainer.test_model(model_path, case.xml_path)
        per_manuscript[case.manuscript_id] = _parse_ketos_stdout(stdout)

    # Aggregate per-style-group: mean of per-manuscript metrics
    sg_metrics: dict[str, float] = {}
    for metric in ("cer", "wer", "accuracy"):
        vals = [m[metric] for m in per_manuscript.values() if metric in m]
        if vals:
            sg_metrics[metric] = sum(vals) / len(vals)

    report = {
        "manifest_id": manifest.manifest_id,
        "style_group_id": style_group_id,
        "script_block": manifest.script_block,
        "model_path": model_path,
        "per_manuscript": per_manuscript,
        "per_style_group": {style_group_id: sg_metrics},
    }

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest.manifest_id}__{style_group_id}__{Path(model_path).stem}"
    (reports / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (reports / f"{stem}.md").write_text(_report_to_markdown(report))
    return report


def _report_to_markdown(report: dict) -> str:
    """Render a benchmark report as a Markdown table."""
    lines = [
        f"# Benchmark: {report['manifest_id']} / {report['style_group_id']}",
        "",
        f"- Script block: `{report['script_block']}`",
        f"- Model: `{report['model_path']}`",
        "",
        "## Per-manuscript",
        "",
        "| Manuscript | CER | WER | Accuracy |",
        "|---|---|---|---|",
    ]
    for ms_id, m in report["per_manuscript"].items():
        lines.append(f"| {ms_id} | {m.get('cer', '—')} | {m.get('wer', '—')} | {m.get('accuracy', '—')} |")
    lines += ["", "## Per-style-group", "",
              "| Style group | CER | WER | Accuracy |",
              "|---|---|---|---|"]
    for sg, m in report["per_style_group"].items():
        lines.append(f"| {sg} | {m.get('cer', '—')} | {m.get('wer', '—')} | {m.get('accuracy', '—')} |")
    return "\n".join(lines) + "\n"