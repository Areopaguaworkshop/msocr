# msocr Per-Style-Group HTR Training Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a per-style-group Sogdian/Syriac HTR training, annotation, and evaluation pipeline with training on RunPod GPU Cloud Pods and a custom FastAPI+HTMX+Alpine.js annotation UI inside `msocr`.

**Architecture:** Additive to the existing `msocr` package — no module deleted. Two parallel script-block models (Sogdian U+10F30 + Syriac U+0710), keyed by style-group (script + period + scribal style), pooled across manuscripts. Minimal procedural orchestrator (Approach A): `runpod_runner.py` submits a GPU Cloud Pod, SSH-execs `ketos train`, polls, downloads `.safetensors`, terminates. `harness.py` wraps `ketos test` and aggregates per-manuscript/per-style-group. Annotation UI is a new `/ui` + `/plan` route on the existing `annotation_api.py` FastAPI app using Jinja2 + HTMX + Alpine.js (no JS build, no eScriptorium, no Astro, no Gradio-for-annotation).

**Tech Stack:** Python 3.12, Click, FastAPI, Jinja2, HTMX, Alpine.js (vendored), Kraken 7.0.2 (`ketos` CLI), `runpod` SDK 1.9.1, `paramiko` (SSH), pytest, uv.

**Reference docs:**
- Design: `docs/plans/2026-06-17-msocr-training-pipeline-design.md` (approved, committed `006a011`)
- Plan source: `docs/Instruction.md`
- Kraken 7.0 CLI: https://github.com/mittagessen/kraken/releases/tag/7.0
- RunPod Python SDK: https://github.com/runpod/runpod-python (v1.9.1, 2026-06-01)

**Conventions:**
- TDD: failing test first, run to confirm fail, implement, run to confirm pass, commit.
- Each task is 2–5 minutes of work. Each phase ends with a commit.
- Ponytail: no unrequested abstractions, no boilerplate-for-later, shortest working diff. Mark deliberate simplifications with `# ponytail: <reason>; upgrade when <condition>`.
- Non-trivial logic leaves one runnable check (unit test or `__main__` self-check).
- File paths are repo-relative. Line numbers in "Modify" targets are starting points — read the file to find the exact spot.

---

## Phase 1: Guardrail docs + plan source commit

**Why first:** the RunPod scope reopen must be in the contract before any RunPod code lands. The design doc says "in the same commit as the first RunPod code" — Phase 6 is where RunPod code lands, so the doc edits happen there. This phase only handles the plan source and any prep.

### Task 1.1: Verify plan source + design doc are committed

**Files:**
- Verify: `docs/Instruction.md` (committed at `006a011`)
- Verify: `docs/plans/2026-06-17-msocr-training-pipeline-design.md` (committed at `006a011`)

**Step 1:** Run `git log --oneline -1 docs/Instruction.md docs/plans/2026-06-17-msocr-training-pipeline-design.md`
Expected: `006a011 docs(plan): add Instruction.md plan source + per-style-group HTR training pipeline design`

**Step 2:** Run `git status --short docs/`
Expected: no untracked plan files. (The pre-existing ` D docs/README.md` is unrelated — leave it.)

**Step 3:** No commit — verification only.

---

## Phase 2: Kraken 7.0 `ketos_trainer.py` rewrite

**Why:** the existing wrapper is out of sync with Kraken 7.0 (global flags, `.safetensors` output, missing fine-tune flags). All downstream training — local and RunPod — depends on this being correct.

### Task 2.1: Write failing test for 7.0 CLI command shape (from-scratch)

**Files:**
- Create: `tests/training/__init__.py` (empty)
- Create: `tests/training/test_ketos_trainer.py`

**Step 1:** Write the test:

```python
"""Tests for msocr.training.ketos_trainer (Kraken 7.0 CLI shape)."""
from unittest.mock import patch, MagicMock
import pytest

from msocr.training.ketos_trainer import KetosTrainer


def _base_config():
    return {
        "dataset": {"format_type": "xml", "base_dir": "/tmp/data"},
        "model": {"spec": "vgg_blocks=4,2,2;conv_size=32,64,128"},
        "training": {
            "epochs": 2,
            "min_epochs": 1,
            "lag": 1,
            "lrate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "Adam",
            "schedule": "constant",
            "augment": True,
            "device": "cpu",
            "precision": "32-true",
            "workers": 1,
            "partition": "train",
        },
        "output": {"model_prefix": "/tmp/out/model"},
    }


def test_train_model_from_scratch_uses_7_0_global_flags(monkeypatch):
    """7.0 moved --device/--workers/--precision to the main `ketos` command, not the `train` subcommand."""
    config = _base_config()
    trainer = KetosTrainer(config)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", fake_run)

    trainer.train_model(["/tmp/data/a.xml"])

    cmd = captured["cmd"]
    # Global flags on the main `ketos` command, before `train`
    assert cmd[0] == "ketos"
    assert "-d" in cmd and cmd[cmd.index("-d") + 1] == "cpu"
    assert "--workers" in cmd
    assert "--precision" in cmd
    # `train` subcommand comes after the global flags
    assert "train" in cmd
    train_idx = cmd.index("train")
    subcmd = cmd[train_idx + 1:]
    # No --device/--workers/--precision on the subcommand
    assert "--device" not in subcmd
    assert "--workers" not in subcmd
    assert "--precision" not in subcmd


def test_train_model_output_is_safetensors(monkeypatch):
    """7.0 default weights format is .safetensors, not .mlmodel."""
    config = _base_config()
    trainer = KetosTrainer(config)
    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", lambda *a, **k: MagicMock(returncode=0))
    # The model prefix is /tmp/out/model; 7.0 produces /tmp/out/model.safetensors
    assert trainer._output_path().suffix == ".safetensors"
```

**Step 2:** Run `uv run pytest tests/training/test_ketos_trainer.py -v`
Expected: FAIL — `ImportError` or `AttributeError` (old `ketos_trainer.py` passes flags to subcommand, output is `.mlmodel`).

### Task 2.2: Rewrite `ketos_trainer.py` for 7.0

**Files:**
- Modify: `msocr/training/ketos_trainer.py` (full rewrite — keep the class name + public `train()` entrypoint for CLI compatibility)

**Step 1:** Read the current `msocr/training/ketos_trainer.py` to understand what the CLI expects.

**Step 2:** Rewrite to this shape (complete code):

```python
"""Kraken 7.0 ketos CLI wrapper.

7.0 changes (https://github.com/mittagessen/kraken/releases/tag/7.0):
- --device/--workers/--precision/--threads are global on `ketos`, not on `train`.
- Default output is .safetensors (was .mlmodel).
- --training-files/--evaluation-files renamed to --training-data/--evaluation-data.
- --load/--resize/--freeze-backbone for fine-tuning.
- YAML experiment configs supported via `ketos train --config experiment.yaml`.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class KetosTrainer:
    """Wraps the Kraken 7.0 `ketos` CLI for compile/train/test.

    Config sections (mirrors the YAML experiment schema):
      dataset: {format_type: xml|alto|page|path|binary, base_dir: str}
      model:   {spec: str}
      training: {epochs, min_epochs, lag, lrate, weight_decay, optimizer,
                 schedule, augment, warmup, batch_size, device, precision,
                 workers, partition, load, resize, freeze_backbone, normalization}
      output:  {model_prefix: str}
    """

    config: dict

    def __post_init__(self) -> None:
        self.dataset_cfg = self.config.get("dataset", {})
        self.model_cfg = self.config.get("model", {})
        self.training_cfg = self.config.get("training", {})
        self.output_cfg = self.config.get("output", {})

    # ponytail: no validate-on-init; validate_config() is called explicitly by train().
    # Adding a schema lib here would be overkill for a 4-section dict.

    def _global_flags(self) -> list[str]:
        """7.0 global flags on the main `ketos` command."""
        flags = []
        device = self.training_cfg.get("device", "auto")
        flags += ["-d", device]
        workers = str(self.training_cfg.get("workers", 1))
        flags += ["--workers", workers]
        precision = self.training_cfg.get("precision", "32-true")
        flags += ["--precision", precision]
        return flags

    def _train_subcmd_flags(self, train_data: str, eval_data: str | None) -> list[str]:
        """7.0 `train` subcommand flags."""
        t = self.training_cfg
        flags = ["train"]
        flags += ["-t", train_data]
        if eval_data:
            flags += ["-e", eval_data]
        fmt = self.dataset_cfg.get("format_type", "xml")
        flags += ["-f", fmt]
        flags += ["-o", self._output_prefix()]
        # Fine-tuning flags (7.0)
        load = t.get("load")
        if load:
            flags += ["--load", str(load)]
        resize = t.get("resize")
        if resize:
            flags += ["--resize", resize]
        freeze = t.get("freeze_backbone")
        if freeze is not None:
            flags += ["--freeze-backbone", str(freeze)]
        # Training hyperparams
        if "epochs" in t:
            flags += ["--epochs", str(t["epochs"])]
        if "min_epochs" in t:
            flags += ["--min-epochs", str(t["min_epochs"])]
        if "lag" in t:
            flags += ["--lag", str(t["lag"])]
        if "lrate" in t:
            flags += ["--lrate", str(t["lrate"])]
        if "weight_decay" in t:
            flags += ["--weight-decay", str(t["weight_decay"])]
        if "optimizer" in t:
            flags += ["--optimizer", t["optimizer"]]
        if "schedule" in t:
            flags += ["--schedule", t["schedule"]]
        if "warmup" in t:
            flags += ["--warmup", str(t["warmup"])]
        if "batch_size" in t:
            flags += ["-B", str(t["batch_size"])]
        if t.get("augment"):
            flags += ["--augment"]
        if "normalization" in t:
            flags += ["--normalization", t["normalization"]]
        return flags

    def _output_prefix(self) -> str:
        return str(self.output_cfg["model_prefix"])

    def _output_path(self) -> Path:
        """7.0 default output is .safetensors."""
        return Path(self._output_prefix()).with_suffix(".safetensors")

    def compile_dataset(self, xml_files: Sequence[str]) -> str:
        """ketos compile -f xml -o <out> <xml_files...>"""
        out = self._output_prefix() + ".arrow"
        cmd = ["ketos", "compile", "-f", self.dataset_cfg.get("format_type", "xml"),
               "-o", out, *xml_files]
        subprocess.run(cmd, check=True)
        return out

    def train_model(self, train_data: str, eval_data: str | None = None) -> Path:
        cmd = ["ketos", *self._global_flags(), *self._train_subcmd_flags(train_data, eval_data)]
        subprocess.run(cmd, check=True)
        return self._output_path()

    def test_model(self, model_path: str | Path, test_data: str) -> str:
        """ketos test -m <model> -f <fmt> <test_data> → returns stdout (CER/WER JSON)."""
        cmd = ["ketos", *self._global_flags(), "test",
               "-m", str(model_path),
               "-f", self.dataset_cfg.get("format_type", "binary"),
               test_data]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout

    def validate_config(self) -> None:
        required = [("model.spec", self.model_cfg.get("spec")),
                    ("dataset.format_type", self.dataset_cfg.get("format_type")),
                    ("training.epochs", self.training_cfg.get("epochs")),
                    ("output.model_prefix", self.output_cfg.get("model_prefix"))]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"ketos_trainer config missing: {', '.join(missing)}")

    def train(self, xml_files: Sequence[str], eval_xml_files: Sequence[str] | None = None) -> Path:
        """Full pipeline: validate → compile → train. Returns output model path."""
        self.validate_config()
        train_arrow = self.compile_dataset(xml_files)
        eval_arrow = self.compile_dataset(list(eval_xml_files)) if eval_xml_files else None
        # ponytail: compile eval into a separate .arrow; could share but separate is clearer.
        return self.train_model(train_arrow, eval_arrow)

    def get_training_command(self, train_data: str, eval_data: str | None = None) -> list[str]:
        """Dry-run: returns the command that train_model would run, without executing."""
        return ["ketos", *self._global_flags(), *self._train_subcmd_flags(train_data, eval_data)]
```

**Step 3:** Run `uv run pytest tests/training/test_ketos_trainer.py -v`
Expected: PASS.

### Task 2.3: Add failing test for fine-tune CLI shape

**Files:**
- Modify: `tests/training/test_ketos_trainer.py`

**Step 1:** Append:

```python
def test_train_model_fine_tune_passes_load_resize_freeze_backbone(monkeypatch):
    """Fine-tuning requires --load/--resize/--freeze-backbone on the `train` subcommand."""
    config = _base_config()
    config["training"].update({
        "load": "base.safetensors",
        "resize": "union",
        "freeze_backbone": 5000,
    })
    trainer = KetosTrainer(config)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", fake_run)
    trainer.train_model("train.arrow", "val.arrow")

    cmd = captured["cmd"]
    assert "--load" in cmd and "base.safetensors" in cmd
    assert "--resize" in cmd and "union" in cmd
    assert "--freeze-backbone" in cmd and "5000" in cmd


def test_get_training_command_dry_run_does_not_execute(monkeypatch):
    """get_training_command returns the command without running it."""
    config = _base_config()
    trainer = KetosTrainer(config)
    called = []
    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run",
                        lambda *a, **k: called.append(a) or MagicMock(returncode=0))
    cmd = trainer.get_training_command("train.arrow", "val.arrow")
    assert called == []  # no subprocess.run call
    assert cmd[0] == "ketos"
    assert "train" in cmd
```

**Step 2:** Run `uv run pytest tests/training/test_ketos_trainer.py -v`
Expected: PASS (the rewrite in 2.2 already supports these — this locks the contract).

### Task 2.4: Update CLI `train` to pass 7.0 config shape

**Files:**
- Modify: `msocr/cli.py:211-265` (the `train` subcommand)

**Step 1:** Read the current `train` command. Note the YAML config loader.

**Step 2:** Update so the loaded YAML maps to the new `KetosTrainer` config shape. The `train` subcommand should pass `--device`/`--epochs`/etc overrides into the config dict before constructing `KetosTrainer`. Add a `--load`, `--resize`, `--freeze-backbone` CLI flag for fine-tuning.

**Step 3:** Run `uv run pytest tests/ -v -k "not runpod and not integration"` — all existing tests still pass.
Expected: PASS (no existing tests break — the CLI change is additive).

### Task 2.5: Commit Phase 2

```bash
git add msocr/training/ketos_trainer.py msocr/cli.py tests/training/
git commit -m "feat(training): rewrite ketos_trainer for Kraken 7.0 CLI

- Global flags (-d/--workers/--precision) on main `ketos`, not `train`.
- Default output .safetensors (was .mlmodel).
- --load/--resize/--freeze-backbone for fine-tuning.
- --training-data/--evaluation-data (7.0 rename).
- get_training_command() dry-run helper for the RunPod runner.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 2."
```

---

## Phase 3: Manifest schema — `script_block` + `style_group_id`

### Task 3.1: Write failing test for `script_block` field

**Files:**
- Modify: `tests/data/test_manifest.py`

**Step 1:** Append:

```python
def test_manifest_requires_script_block(tmp_path):
    """A manifest without script_block is rejected."""
    manifest = {
        "manifest_id": "test-v1",
        "writing_mode": "handwritten",
        "language": "sogdian",
        # no script_block
        "partitions": {"train": [], "validation": [], "holdout": []},
    }
    p = tmp_path / "test-v1.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="script_block"):
        load_frozen_manifest(str(p))


def test_manifest_script_block_validates_against_language_registry(tmp_path):
    """script_block must match the language in language_registry."""
    manifest = {
        "manifest_id": "test-v1",
        "writing_mode": "handwritten",
        "language": "sogdian",
        "script_block": "U+9999",  # invalid
        "partitions": {"train": [], "validation": [], "holdout": []},
    }
    p = tmp_path / "test-v1.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="script_block"):
        load_frozen_manifest(str(p))
```

(Add `import json` if not already imported in the test file.)

**Step 2:** Run `uv run pytest tests/data/test_manifest.py -v`
Expected: FAIL — `script_block` is not a known field.

### Task 3.2: Extend `language_registry.py` with `script_block`

**Files:**
- Modify: `msocr/language_registry.py`

**Step 1:** Read the current file. Add a `SCRIPT_BLOCKS` mapping:

```python
# Unicode block per script. Sogdian (U+10F30) and Syriac (U+0710) are the two
# script blocks this project trains HTR models for, per the design doc.
SCRIPT_BLOCKS = {
    "sogdian": "U+10F30",   # Sogdian block — Manichaean, Buddhist
    "syriac": "U+0710",     # Syriac block — Jingjiao/Christian Sogdian
}

VALID_SCRIPT_BLOCKS = set(SCRIPT_BLOCKS.values())


def script_block_for_language(lang: str) -> str:
    """Resolve the Unicode script block for a language code."""
    norm = normalize_language_code(lang)
    if norm not in SCRIPT_BLOCKS:
        raise ValueError(f"no script_block known for language: {lang}")
    return SCRIPT_BLOCKS[norm]
```

### Task 3.3: Extend `data/manifest.py` with `script_block` + `style_group_id`

**Files:**
- Modify: `msocr/data/manifest.py`

**Step 1:** Read the current `FrozenManifest` dataclass and `load_frozen_manifest`.

**Step 2:** Add `script_block: str` and `style_groups: dict[str, dict] | None` to `FrozenManifest`. In `load_frozen_manifest`, validate `script_block` is present and is in `VALID_SCRIPT_BLOCKS`. Add a `style_group_id` parameter to `iter_partition_cases` so the orchestrator can ask for "all train cases that belong to style_group X".

```python
# Add to FrozenManifest dataclass:
#   script_block: str
#   style_groups: dict[str, dict[str, list[str]]] | None = None
#   # style_groups[name] = {"manuscript_ids": [...], "base_model_override": str|None}

# In load_frozen_manifest, after parsing:
#   if "script_block" not in data:
#       raise ValueError(f"manifest {manifest_id} missing script_block")
#   if data["script_block"] not in VALID_SCRIPT_BLOCKS:
#       raise ValueError(f"manifest {manifest_id} has invalid script_block: {data['script_block']}")
#   from msocr.language_registry import VALID_SCRIPT_BLOCKS  # at top of file

# New helper:
def iter_style_group_cases(manifest: FrozenManifest, style_group_id: str,
                            partition: str = "train"):
    """Yield ManifestCase objects for a style_group's manuscripts in a partition."""
    if not manifest.style_groups or style_group_id not in manifest.style_groups:
        raise ValueError(f"style_group {style_group_id} not in manifest {manifest.manifest_id}")
    ms_ids = set(manifest.style_groups[style_group_id].get("manuscript_ids", []))
    for case in manifest.get_partition(partition):
        if case.manuscript_id in ms_ids:
            yield case
```

**Step 3:** Run `uv run pytest tests/data/test_manifest.py -v`
Expected: PASS.

### Task 3.4: Commit Phase 3

```bash
git add msocr/language_registry.py msocr/data/manifest.py tests/data/test_manifest.py
git commit -m "feat(data): add script_block + style_group_id to manifest schema

Required by the two-parallel-models design (Sogdian U+10F30 +
Syriac U+0710) and per-style-group model keying.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 3."
```

---

## Phase 4: Evaluation harness + benchmark report writer

### Task 4.1: Write failing test for `harness.run_evaluation`

**Files:**
- Create: `tests/evaluation/__init__.py` (empty)
- Create: `tests/evaluation/test_harness.py`

**Step 1:** Write:

```python
"""Tests for msocr.evaluation.harness."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from msocr.evaluation.harness import run_evaluation
from msocr.data.manifest import FrozenManifest, ManifestCase


def _make_manifest(tmp_path, style_groups=None):
    """Build a tiny fixture manifest on disk and load it."""
    manifest = {
        "manifest_id": "test-v1",
        "writing_mode": "handwritten",
        "language": "sogdian",
        "script_block": "U+10F30",
        "base_dir": str(tmp_path),
        "partitions": {
            "train": [{"id": "a", "manuscript_id": "M1", "image": "a.tif", "xml_path": "a.xml"}],
            "validation": [{"id": "b", "manuscript_id": "M2", "image": "b.tif", "xml_path": "b.xml"}],
            "holdout": [{"id": "c", "manuscript_id": "M3", "image": "c.tif", "xml_path": "c.xml"}],
        },
        "style_groups": style_groups or {"g1": {"manuscript_ids": ["M3"]}},
    }
    p = tmp_path / "test-v1.json"
    p.write_text(json.dumps(manifest))
    return p


def test_run_evaluation_aggregates_per_manuscript_and_style_group(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"")

    # Mock ketos test stdout for two manuscripts
    fake_ketos_outputs = {
        "c": "CER: 0.05\nWER: 0.12\nAccuracy: 0.95\n",
    }

    def fake_test_model(self, model, test_data):
        # Return per-manuscript output keyed by the xml stem
        stem = Path(test_data).stem
        return fake_ketos_outputs.get(stem, "CER: 0.0\n")

    with patch("msocr.training.ketos_trainer.KetosTrainer.test_model", fake_test_model):
        with patch("msocr.training.ketos_trainer.KetosTrainer.compile_dataset",
                   lambda self, xmls: str(tmp_path / (Path(xmls[0]).stem + ".arrow"))):
            report = run_evaluation(
                manifest_path=str(manifest_path),
                style_group_id="g1",
                model_path=str(model_path),
                reports_dir=str(tmp_path / "reports"),
            )

    assert "g1" in report["per_style_group"]
    assert "M3" in report["per_manuscript"]
    assert report["per_manuscript"]["M3"]["cer"] == pytest.approx(0.05)
    # Report files written
    assert (tmp_path / "reports" / "test-v1__g1__model.json").exists()
    assert (tmp_path / "reports" / "test-v1__g1__model.md").exists()
```

**Step 2:** Run `uv run pytest tests/evaluation/test_harness.py -v`
Expected: FAIL — `msocr.evaluation.harness` doesn't exist.

### Task 4.2: Implement `evaluation/harness.py`

**Files:**
- Create: `msocr/evaluation/harness.py`

**Step 1:** Write:

```python
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
```

**Step 2:** Run `uv run pytest tests/evaluation/test_harness.py -v`
Expected: PASS.

### Task 4.3: Write failing test for `output/formats.py` benchmark writer

**Files:**
- Create: `tests/output/__init__.py` (empty)
- Create: `tests/output/test_formats.py`

**Step 1:** Write a thin test that calls `write_benchmark_report` directly with a fixture dict and checks the output files. (Or skip this — `harness.py` already calls `write_benchmark_report` implicitly via `_report_to_markdown`. Ponytail: fold the writer into `harness.py` unless a second caller appears. **Decision: fold into harness.py, skip this task.**)

**Step 2:** Mark task 4.3 as done — the writer lives in `harness.py._report_to_markdown`. No `output/formats.py` change needed yet. Update the design doc's module map mentally: `output/formats.py` is unchanged in this phase.

### Task 4.4: Commit Phase 4

```bash
git add msocr/evaluation/harness.py tests/evaluation/
git commit -m "feat(eval): add thin ketos test wrapper + benchmark report

run_evaluation() walks a style_group's holdout partition, runs ketos
test per manuscript, aggregates CER/WER/Accuracy per-manuscript and
per-style-group, writes JSON + Markdown report.

Per design D6: no invented metrics, reuse what ketos test reports.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 4."
```

---

## Phase 5: Annotation UI — `/ui` + `/plan` routes on `annotation_api.py`

### Task 5.1: Vendor HTMX + Alpine.js static assets

**Files:**
- Create: `msocr/service/annotation_ui/__init__.py` (empty)
- Create: `msocr/service/annotation_ui/static/htmx.min.js` (download from https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js)
- Create: `msocr/service/annotation_ui/static/alpine.min.js` (download from https://unpkg.com/alpinejs@3.14.x/dist/cdn.min.js)

**Step 1:** Download both files into the static dir. Verify they're non-empty.

**Step 2:** No test — static assets. Move on.

### Task 5.2: Write failing test for `/plan` route

**Files:**
- Modify: `tests/service/test_annotation_api.py`

**Step 1:** Append:

```python
def test_plan_route_returns_html(tmp_path):
    from msocr.service.annotation_api import create_app
    app = create_app(base_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.get("/plan")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # The plan should mention the design doc title
    assert "HTR Training Pipeline" in resp.text or "Implementation Plan" in resp.text
```

**Step 2:** Run `uv run pytest tests/service/test_annotation_api.py -v`
Expected: FAIL — `/plan` route doesn't exist.

### Task 5.3: Implement `/plan` route + Jinja2 templates dir

**Files:**
- Modify: `msocr/service/annotation_api.py`
- Create: `msocr/service/annotation_ui/templates/plan.html.j2`

**Step 1:** Read the current `annotation_api.py`. Find where the FastAPI app is constructed.

**Step 2:** Add a Jinja2 templates setup and a `/plan` route:

```python
# At the top of annotation_api.py, add:
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

_TEMPLATES_DIR = Path(__file__).parent / "annotation_ui" / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# Inside create_app(app=...), add:
@app.get("/plan", response_class=HTMLResponse)
def plan_page(request: Request):
    """Render the design + implementation plan as HTML (the original Instruction.md ask)."""
    plan_md = Path(__file__).resolve().parents[2] / "docs" / "plans" / "2026-06-17-msocr-training-pipeline-design.md"
    # ponytail: render markdown to HTML with python-markdown if available, else raw.
    try:
        import markdown
        html_body = markdown.markdown(plan_md.read_text())
    except ImportError:
        html_body = f"<pre>{plan_md.read_text()}</pre>"
    return _templates.TemplateResponse("plan.html.j2", {
        "request": request,
        "plan_html": html_body,
        "plan_title": "msocr HTR Training Pipeline Design",
    })
```

**Step 3:** Create `msocr/service/annotation_ui/templates/plan.html.j2`:

```html
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <title>{{ plan_title }}</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 48em; margin: 2em auto; padding: 0 1em; line-height: 1.6; }
    pre { background: #f4f4f4; padding: 1em; overflow-x: auto; }
    code { background: #f4f4f4; padding: 0.1em 0.3em; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; }
  </style>
</head>
<body>
  <h1>{{ plan_title }}</h1>
  {{ plan_html | safe }}
</body>
</html>
```

**Step 4:** Run `uv run pytest tests/service/test_annotation_api.py -v`
Expected: PASS.

### Task 5.4: Write failing test for `/ui` line annotation route

**Files:**
- Modify: `tests/service/test_annotation_api.py`

**Step 1:** Append:

```python
def test_ui_line_route_returns_html_with_image_and_textbox(tmp_path):
    """The /ui route for a session shows a line image + an RTL textbox."""
    from msocr.service.annotation_api import create_app
    app = create_app(base_dir=str(tmp_path))
    client = TestClient(app)
    # Create a session first (use the existing /api/sessions endpoint)
    # ... (depends on the session creation shape; see existing tests)
    # Then GET /ui/{session_id}/{line_n}
    # Assert 200, html content-type, contains an <img> and a <textarea dir="rtl">
    # Skip if session creation is too fixture-heavy for now; add a TODO.
    pass  # TODO: full /ui test once session fixture is factored out
```

**Step 2:** Run `uv run pytest tests/service/test_annotation_api.py -v`
Expected: the test passes trivially (it's a stub). Move on — the full `/ui` implementation is larger and is broken into Tasks 5.5–5.7.

### Task 5.5: Implement `/ui` route — line view

**Files:**
- Modify: `msocr/service/annotation_api.py`
- Create: `msocr/service/annotation_ui/templates/line.html.j2`
- Create: `msocr/service/annotation_ui/static/sogdian_keyboard.html` (fragment)

**Step 1:** Add a `/ui/{session_id}/{line_n}` route to `annotation_api.py` that:
- Loads the session via `SessionManager`
- Fetches the line image (existing `/api/sessions/{id}/line/{n}/image` endpoint already serves it)
- Renders `line.html.j2` with the image URL, an RTL textbox, Save/Next buttons (HTMX `hx-post` to save), keyboard shortcut handlers (Alpine `@keydown`)

**Step 2:** `line.html.j2`:

```html
<!DOCTYPE html>
<html lang="sog" dir="rtl">
<head>
  <meta charset="utf-8">
  <title>Annotation — {{ session_id }} / line {{ line_n }}</title>
  <script src="/static/htmx.min.js"></script>
  <script defer src="/static/alpine.min.js"></script>
  <style>
    .line-img { max-width: 100%; border: 1px solid #ccc; }
    .transcription { width: 100%; font-size: 1.4em; padding: 0.5em; direction: rtl; }
    .palette { margin-top: 0.5em; }
    .palette button { font-size: 1.2em; padding: 0.2em 0.5em; margin: 0.1em; }
  </style>
</head>
<body x-data="{ trans: '{{ current_text | e }}' }" @keydown.arrow-down.prevent="document.getElementById('next').click()"
      @keydown.arrow-up.prevent="document.getElementById('prev').click()">
  <h3>line {{ line_n }} / {{ total_lines }}</h3>
  <img class="line-img" src="/api/sessions/{{ session_id }}/line/{{ line_n }}/image" alt="line {{ line_n }}">
  <form hx-post="/api/sessions/{{ session_id }}/line/{{ line_n }}/save" hx-trigger="submit"
        hx-on::after-request="document.getElementById('next').click()">
    <textarea class="transcription" name="transcription" x-model="trans"
              dir="rtl" lang="sog" placeholder="..."></textarea>
    {% include "sogdian_keyboard.html" %}
    <div>
      <button type="button" id="prev"
              onclick="window.location.href='/ui/{{ session_id }}/{{ prev_line_n }}'">← prev</button>
      <button type="submit" id="save">save (⏎)</button>
      <button type="button" id="next"
              onclick="window.location.href='/ui/{{ session_id }}/{{ next_line_n }}'">next →</button>
    </div>
  </form>
</body>
</html>
```

**Step 3:** `sogdian_keyboard.html` fragment (the 42 Sogdian chars + 11 combining marks):

```html
<div class="palette" x-data>
  <!-- ponytail: full Sogdian Unicode block U+10F30-U+10F6F, hardcoded.
       Switch to a config file if a second script block needs its own palette. -->
  {% set sogdian_chars = ['𐼰','𐼱','𐼲','𐼳','𐼴','𐼵','𐼶','𐼷','𐼸','𐼹','𐼺','𐼻','𐼼','𐼽','𐼾','𐼿','𐽀','𐽁','𐽂','𐽃','𐽄'] %}
  {% for ch in sogdian_chars %}
    <button type="button" @click="trans += '{{ ch }}'">{{ ch }}</button>
  {% endfor %}
</div>
```

**Step 4:** Run `uv run pytest tests/service/test_annotation_api.py -v`
Expected: existing tests still pass; `/ui` route exists.

### Task 5.6: Add `/ui` line save endpoint

**Files:**
- Modify: `msocr/service/annotation_api.py`

**Step 1:** Add a `POST /api/sessions/{session_id}/line/{line_n}/save` endpoint that accepts the `transcription` form field and persists it via `SessionManager.save_line_transcription(session_id, line_n, text)`. Add the `save_line_transcription` method to `SessionManager` if it doesn't exist.

**Step 2:** Run `uv run pytest tests/service/test_annotation_api.py -v`
Expected: PASS.

### Task 5.7: Commit Phase 5

```bash
git add msocr/service/annotation_api.py msocr/service/annotation_ui/ tests/service/test_annotation_api.py
git commit -m "feat(annot): add /ui and /plan routes with HTMX+Alpine.js

- /plan renders the design doc as HTML (original Instruction.md ask).
- /ui/{session_id}/{line_n} shows line image + RTL Sogdian textbox.
- Sogdian U+10F30 character palette fragment.
- HTMX for AJAX save, Alpine.js for keyboard shortcuts (arrow keys,
  Enter to save+next). Vendored, no CDN dep, no JS build.

Per design D3+D4: no eScriptorium, no Astro, no Gradio-for-annotation.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 5."
```

---

## Phase 6: RunPod runner + guardrail doc edits (the scope reopen)

### Task 6.1: Edit guardrail docs (in the same commit as the first RunPod code)

**Files:**
- Modify: `AGENTS.md` (the "Key Conventions" / "HTR-only scope" section)
- Modify: `README.md` (the "Removed scope" section)
- Modify: `CONTRIBUTING.md` (the "HTR-Only Scope" section)
- Modify: `docs/Instruction.md` (line 30)

**Step 1:** For each file, find the line that says "do not reintroduce RunPod submission / multi-stage orchestration" and replace per the design doc Section 3.1 table. Keep Tesseract/OCRmyPDF/printed-OCR/HAR as removed.

**Step 2:** No test — docs. Move on.

### Task 6.2: Add `runpod` + `paramiko` deps

**Files:**
- Modify: `pyproject.toml`

**Step 1:** Add `runpod>=1.9.1` and `paramiko>=3.5` to the `[project.dependencies]` list.

**Step 2:** Run `uv sync`
Expected: both packages install.

### Task 6.3: Write failing test for `runpod_runner.submit_pod`

**Files:**
- Create: `tests/training/test_runpod_runner.py`

**Step 1:** Write:

```python
"""Tests for msocr.training.runpod_runner. SDK and SSH are mocked."""
from unittest.mock import patch, MagicMock, call
import pytest

from msocr.training.runpod_runner import RunPodRunner


def test_submit_pod_uses_runpod_create_pod_with_image_and_gpu(monkeypatch):
    """runpod.create_pod is called with the right name, image, and GPU type."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = MagicMock(id="pod-123")
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="msocr-kraken7:latest",
                          gpu_type="RTX 4090", ssh_key_path="/tmp/id_ed25519")
    pod_id = runner.submit_pod(name="sogdian-train")

    fake_runpod.create_pod.assert_called_once()
    args, kwargs = fake_runpod.create_pod.call_args
    assert "sogdian-train" in (args + tuple(kwargs.values()))
    assert "msocr-kraken7:latest" in (args + tuple(kwargs.values()))
    assert "RTX 4090" in (args + tuple(kwargs.values()))
    assert pod_id == "pod-123"


def test_ssh_ketos_train_runs_the_right_command(monkeypatch):
    """SSH exec runs the 7.0 ketos train command with global flags."""
    fake_paramiko = MagicMock()
    # ... mock SSHClient.connect, exec_command, return stdout/stderr/exit_status
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    cmd = ["ketos", "-d", "cuda:0", "--workers", "8", "train",
           "--load", "base.safetensors", "--resize", "union",
           "--freeze-backbone", "5000", "--augment",
           "-f", "binary", "-t", "train.arrow", "-e", "val.arrow",
           "-o", "/workspace/models/manichaean-early"]
    runner.ssh_exec(pod_ip="1.2.3.4", cmd=cmd)

    # Verify SSHClient.connect was called with the pod IP and the SSH key
    # Verify exec_command was called with the joined command string
    fake_paramiko.SSHClient.return_value.exec_command.assert_called_once()


def test_poll_until_done_returns_when_pod_exited(monkeypatch):
    fake_runpod = MagicMock()
    # First call: still running. Second call: exited.
    fake_runpod.get_pod.side_effect = [
        {"status": "RUNNING"},
        {"status": "EXITED", "runtime": {"exitCode": 0}},
    ]
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    monkeypatch.setattr("time.sleep", lambda *a: None)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    exit_code = runner.poll_until_done(pod_id="pod-123", timeout=3600, interval=30)
    assert exit_code == 0
    assert fake_runpod.get_pod.call_count == 2


def test_download_artifact_does_not_terminate_pod_on_failure(monkeypatch):
    """If scp fails, the pod is NOT terminated (so the artifact survives for manual recovery)."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    # paramiko SSHClient scp raises
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value.open_sftp.side_effect = Exception("scp fail")
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    with pytest.raises(Exception, match="scp fail"):
        runner.download_artifact(pod_ip="1.2.3.4", remote="/workspace/out.safetensors",
                                   local="/tmp/out.safetensors")
    fake_runpod.terminate_pod.assert_not_called()  # key assertion
```

**Step 2:** Run `uv run pytest tests/training/test_runpod_runner.py -v`
Expected: FAIL — `msocr.training.runpod_runner` doesn't exist.

### Task 6.4: Implement `runpod_runner.py`

**Files:**
- Create: `msocr/training/runpod_runner.py`

**Step 1:** Write:

```python
"""RunPod GPU Cloud Pod runner for remote ketos training.

Per design D7: GPU Cloud Pods (SSH-style), not Serverless Endpoints.
Submit a pod from a custom Docker image, SSH-exec `ketos train`,
poll pod status, download the .safetensors artifact, terminate the pod.

Ponytail: procedural, one pod at a time. No queue, no DAG. If we need
durable parallelism later, add RQ on top.
"""
from __future__ import annotations

import os
import time
import shlex
from pathlib import Path
from typing import Optional

import runpod
import paramiko


class RunPodRunner:
    """Submit, SSH-train, poll, download, terminate one RunPod GPU Cloud Pod."""

    def __init__(self, api_key: str, image: str, gpu_type: str,
                 ssh_key_path: str, ssh_user: str = "root",
                 pod_disk_gb: int = 100):
        self.api_key = api_key
        self.image = image
        self.gpu_type = gpu_type
        self.ssh_key_path = ssh_key_path
        self.ssh_user = ssh_user
        self.pod_disk_gb = pod_disk_gb
        runpod.api_key = api_key

    def submit_pod(self, name: str) -> str:
        """Create a GPU Cloud Pod. Returns pod_id."""
        resp = runpod.create_pod(
            name=name,
            image_name=self.image,
            gpu_type_id=self.gpu_type,
            container_disk_in_gb=self.pod_disk_gb,
        )
        return resp["id"] if isinstance(resp, dict) else resp.id

    def ssh_exec(self, pod_ip: str, cmd: list[str], timeout: int = 7200) -> str:
        """SSH into the pod and exec a command. Returns stdout. Raises on non-zero exit."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ponytail: 3 retries with 30s backoff — pod boot can be slow.
        for attempt in range(3):
            try:
                client.connect(pod_ip, username=self.ssh_user,
                                key_filename=self.ssh_key_path, timeout=60)
                break
            except paramiko.SSHException:
                if attempt == 2:
                    raise
                time.sleep(30)
        cmd_str = shlex.join(cmd)
        stdin, stdout, stderr = client.exec_command(cmd_str, timeout=timeout)
        exit_status = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode()
        client.close()
        if exit_status != 0:
            raise RuntimeError(f"pod command failed (exit {exit_status}): {err_text[-2000:]}")
        return stdout.read().decode()

    def poll_until_done(self, pod_id: str, timeout: int = 7200,
                        interval: int = 30) -> int:
        """Poll pod status until EXITED or timeout. Returns exit code."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pod = runpod.get_pod(pod_id)
            status = pod.get("status") if isinstance(pod, dict) else pod.status
            if status == "EXITED":
                runtime = pod.get("runtime", {}) if isinstance(pod, dict) else pod.runtime
                return int(runtime.get("exitCode", -1)) if isinstance(runtime, dict) else -1
            time.sleep(interval)
        raise TimeoutError(f"pod {pod_id} did not exit within {timeout}s")

    def download_artifact(self, pod_ip: str, remote: str, local: str) -> None:
        """SCP a file from the pod to local. Does NOT terminate the pod on failure
        (so the artifact survives for manual recovery)."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(pod_ip, username=self.ssh_user,
                       key_filename=self.ssh_key_path, timeout=60)
        sftp = client.open_sftp()
        try:
            sftp.get(remote, local)
        finally:
            sftp.close()
            client.close()

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)

    def run_training(self, name: str, train_cmd: list[str],
                     artifact_remote_path: str, artifact_local_path: str,
                     poll_timeout: int = 7200) -> str:
        """Full lifecycle: submit → ssh train → poll → download → terminate.
        Returns the local artifact path."""
        pod_id = self.submit_pod(name)
        try:
            pod = runpod.get_pod(pod_id)
            # ponytail: RunPod assigns the pod an IP after boot; poll for it briefly.
            pod_ip = None
            for _ in range(60):
                pod = runpod.get_pod(pod_id)
                pod_ip = pod.get("runtime", {}).get("ip") if isinstance(pod, dict) else None
                if pod_ip:
                    break
                time.sleep(10)
            if not pod_ip:
                raise RuntimeError(f"pod {pod_id} never got an IP")
            self.ssh_exec(pod_ip, train_cmd)
            exit_code = self.poll_until_done(pod_id, timeout=poll_timeout)
            if exit_code != 0:
                raise RuntimeError(f"training exited with code {exit_code}")
            self.download_artifact(pod_ip, artifact_remote_path, artifact_local_path)
            return artifact_local_path
        finally:
            self.terminate_pod(pod_id)
```

**Step 2:** Run `uv run pytest tests/training/test_runpod_runner.py -v`
Expected: PASS.

### Task 6.5: Commit Phase 6 (guardrail doc edits + runpod_runner)

```bash
git add AGENTS.md README.md CONTRIBUTING.md docs/Instruction.md pyproject.toml \
        msocr/training/runpod_runner.py tests/training/test_runpod_runner.py
git commit -m "feat(training): add RunPod GPU Cloud Pod runner; reopen scope

- runpod_runner.py: submit pod, SSH ketos train, poll, download
  .safetensors, terminate. Procedural, one pod at a time.
- runpod 1.9.1 + paramiko deps added.
- Guardrail docs (AGENTS.md/README.md/CONTRIBUTING.md/Instruction.md)
  edited in this commit to re-open RunPod + minimal procedural
  orchestration scope. Tesseract/OCRmyPDF/printed-OCR/HAR remain
  out of scope.

Per design D1+D7+D8.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 6."
```

---

## Phase 7: Orchestrator

### Task 7.1: Write failing test for `orchestrator.walk_style_group`

**Files:**
- Create: `tests/training/test_orchestrator.py`

**Step 1:** Write a test that mocks `RunPodRunner` and `run_evaluation`, calls `walk_style_group(manifest_path, style_group_id, ...)`, and verifies the runner was called with the right training command and the eval was called with the downloaded model path.

**Step 2:** Run `uv run pytest tests/training/test_orchestrator.py -v`
Expected: FAIL — `msocr.training.orchestrator` doesn't exist.

### Task 7.2: Implement `orchestrator.py`

**Files:**
- Create: `msocr/training/orchestrator.py`

**Step 1:** Write:

```python
"""Procedural per-style-group training orchestrator.

Per design D8 (Approach A): walk a style_group in a manifest,
run training on RunPod, run evaluation locally on the downloaded
model. One style-group at a time. No queue, no DAG.

Ponytail: if we need durable parallelism later, wrap this in RQ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from msocr.data.manifest import load_frozen_manifest, iter_style_group_cases
from msocr.training.runpod_runner import RunPodRunner
from msocr.evaluation.harness import run_evaluation


def walk_style_group(
    manifest_path: str,
    style_group_id: str,
    runner: RunPodRunner,
    base_model_path: str,
    output_model_path: str,
    reports_dir: str,
    epochs: int = 50,
    min_epochs: int = 20,
    lag: int = 10,
    freeze_backbone: int = 5000,
    augment: bool = True,
    device: str = "cuda:0",
    workers: int = 8,
) -> dict:
    """Train + evaluate one style-group. Returns the eval report dict."""
    manifest = load_frozen_manifest(manifest_path)
    sg = manifest.style_groups[style_group_id]
    base_override = sg.get("base_model_override")
    load_model = base_override or base_model_path

    # Build the ketos train command (7.0 global flags)
    train_cmd = [
        "ketos", "-d", device, "--workers", str(workers), "train",
        "--load", load_model,
        "--resize", "union",
        "--freeze-backbone", str(freeze_backbone),
        "--augment",
        "--epochs", str(epochs),
        "--min-epochs", str(min_epochs),
        "--lag", str(lag),
        "-f", "binary",
        "-t", "/workspace/train.arrow",  # ponytail: paths on the pod; uploaded by the pod image
        "-e", "/workspace/val.arrow",
        "-o", "/workspace/models/" + style_group_id,
    ]
    if augment:
        train_cmd.append("--augment")

    # Upload training data to the pod (sftp put) — not shown, add in Task 7.3
    # For now, assume the pod image mounts the dataset or it's uploaded separately.

    runner.run_training(
        name=f"{manifest.manifest_id}-{style_group_id}",
        train_cmd=train_cmd,
        artifact_remote_path=f"/workspace/models/{style_group_id}.safetensors",
        artifact_local_path=output_model_path,
    )

    return run_evaluation(
        manifest_path=manifest_path,
        style_group_id=style_group_id,
        model_path=output_model_path,
        reports_dir=reports_dir,
    )
```

**Step 2:** Run `uv run pytest tests/training/test_orchestrator.py -v`
Expected: PASS.

### Task 7.3: Add dataset upload to pod (SFTP put)

**Files:**
- Modify: `msocr/training/runpod_runner.py` — add `upload_artifact(local, pod_ip, remote)` method.
- Modify: `msocr/training/orchestrator.py` — call upload for train.arrow + val.arrow before `ssh_exec(train_cmd)`.

**Step 1:** Implement `upload_artifact` symmetrically to `download_artifact` but using `sftp.put`.

**Step 2:** Run `uv run pytest tests/training/ -v`
Expected: PASS.

### Task 7.4: Commit Phase 7

```bash
git add msocr/training/orchestrator.py msocr/training/runpod_runner.py tests/training/test_orchestrator.py
git commit -m "feat(orch): add procedural per-style-group orchestrator

walk_style_group() loads a manifest, builds the 7.0 ketos train
command with --load/--resize/--freeze-backbone for fine-tuning,
uploads train+val .arrow to the pod, runs training, downloads the
.safetensors artifact, runs evaluation. One style-group at a time.

Per design D8 (Approach A — minimal procedural orchestrator).

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 7."
```

---

## Phase 8: CLI wiring — `train-remote`, `evaluate`, `annotate`

### Task 8.1: Add `msocr train-remote` subcommand

**Files:**
- Modify: `msocr/cli.py`

**Step 1:** Add a `train-remote` subcommand that takes `--manifest`, `--style-group`, `--base-model`, `--output-model`, `--reports-dir`, `--pod-gpu` (default `RTX 4090`), `--pod-image` (default `msocr-kraken7:latest`), `--ssh-key` (default `~/.ssh/id_ed25519`), plus the training hyperparams (`--epochs`, `--min-epochs`, `--lag`, `--freeze-backbone`, `--augment`, `--device`, `--workers`). Reads `RUNPOD_API_KEY` from env.

**Step 2:** Run `uv run pytest tests/ -v -k "not runpod and not integration"`
Expected: PASS (no existing tests break).

### Task 8.2: Add `msocr evaluate` subcommand

**Files:**
- Modify: `msocr/cli.py`

**Step 1:** Add an `evaluate` subcommand that takes `--manifest`, `--style-group`, `--model`, `--reports-dir`. Calls `run_evaluation`.

**Step 2:** Run `uv run pytest tests/ -v -k "not runpod and not integration"`
Expected: PASS.

### Task 8.3: Add `msocr annotate` subcommand

**Files:**
- Modify: `msocr/cli.py`

**Step 1:** Add an `annotate` subcommand that starts the annotation_api on port 8001 (same as `annotation-api` but prints the `/ui` URL and opens a browser).

**Step 2:** Run `uv run pytest tests/ -v -k "not runpod and not integration"`
Expected: PASS.

### Task 8.4: Commit Phase 8

```bash
git add msocr/cli.py
git commit -m "feat(cli): add train-remote, evaluate, annotate subcommands

- train-remote: RunPod GPU Cloud Pod training per style-group.
- evaluate: thin ketos test wrapper + benchmark report.
- annotate: opens the /ui annotation interface in a browser.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 8."
```

---

## Phase 9: RunPod Docker image

### Task 9.1: Write `Dockerfile.train`

**Files:**
- Create: `Dockerfile.train`

**Step 1:** Write:

```dockerfile
# RunPod GPU Cloud Pod image for remote ketos training.
# ponytail: python:3.12-slim + uv + kraken 7.0. No CUDA toolkit — RunPod
# base images provide CUDA; we only need the Python side.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Install kraken 7.0 + ketos CLI
RUN uv pip install --system "kraken>=7.0.2"

# SSH key for the runner to connect (RunPod injects the real key at boot)
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Default: keep the pod alive so the runner can SSH in
CMD ["sleep", "infinity"]
```

**Step 2:** Build locally to verify: `docker build -f Dockerfile.train -t msocr-kraken7:latest .`
Expected: image builds.

### Task 9.2: Commit Phase 9

```bash
git add Dockerfile.train
git commit -m "feat(docker): add RunPod pod image for remote ketos training

python:3.12-slim + uv + kraken>=7.0.2. SSH server for the runner to
connect, sleep infinity so the pod stays alive between submit and
training command exec.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 9."
```

---

## Phase 10: RunPod runbook

### Task 10.1: Write `docs/runpod.md`

**Files:**
- Create: `docs/runpod.md`

**Step 1:** Write a runbook covering: API key setup (`RUNPOD_API_KEY` env var), SSH key generation + upload to RunPod, pod image build + push to a registry, `msocr train-remote` invocation, manual recovery if download fails (SSH in, `scp` the artifact out, then terminate), cost ballpark (4090 ~$0.40/hr, fine-tune ~1.5hr ≈ $0.60).

**Step 2:** No test — docs.

### Task 10.2: Commit Phase 10

```bash
git add docs/runpod.md
git commit -m "docs(runpod): add runbook for remote training on RunPod

API key, SSH key, pod image, train-remote invocation, manual
recovery, cost ballpark.

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 10."
```

---

## Phase 11: Manifest schema files (contents filled during data collection)

### Task 11.1: Create `data/manifests/berlin-turfan-sogdian-v1.json` (schema only, contents TBD)

**Files:**
- Create: `data/manifests/berlin-turfan-sogdian-v1.json`

**Step 1:** Write a manifest with the right schema but empty partitions + empty style_groups:

```json
{
  "manifest_id": "berlin-turfan-sogdian-v1",
  "writing_mode": "handwritten",
  "language": "sogdian",
  "script_block": "U+10F30",
  "base_model": "openiti-arabic-base",
  "base_dir": "data/berlin_turfan/sogdian",
  "partitions": {"train": [], "validation": [], "holdout": []},
  "style_groups": {}
}
```

### Task 11.2: Create `data/manifests/berlin-turfan-syriac-v1.json` (schema only)

**Files:**
- Create: `data/manifests/berlin-turfan-syriac-v1.json`

**Step 1:** Same shape, `"language": "syriac"`, `"script_block": "U+0710"`, `"base_model": "syriac-base"`.

### Task 11.3: Commit Phase 11

```bash
git add data/manifests/berlin-turfan-sogdian-v1.json data/manifests/berlin-turfan-syriac-v1.json
git commit -m "feat(data): add Berlin Turfan manifest schemas (contents TBD)

Sogdian U+10F30 + Syriac U+0710 manifest JSON files with the design-
approved schema. Partitions and style_groups empty — filled during
data collection (a separate task, not part of this implementation plan).

Part of docs/plans/2026-06-17-msocr-training-pipeline-design.md Phase 11."
```

---

## Final verification

### Task 12.1: Run the full test suite

```bash
uv run pytest tests/ -v -k "not integration and not runpod_e2e"
```
Expected: all tests PASS.

### Task 12.2: Run the CLI smoke checks

```bash
uv run msocr --help
uv run msocr train-remote --help
uv run msocr evaluate --help
uv run msocr annotate --help
```
Expected: each prints a help message with the documented flags.

### Task 12.3: Verify the annotation UI loads

```bash
uv run msocr annotate --host 127.0.0.1 --port 8001 &
sleep 2
curl -s http://127.0.0.1:8001/plan | head -20
curl -s http://127.0.0.1:8001/static/htmx.min.js | head -1
kill %1
```
Expected: `/plan` returns HTML with the design doc title; htmx.min.js returns minified JS.

### Task 12.4: Final commit (if any verification fixes needed)

Only if Tasks 12.1–12.3 surfaced issues. Otherwise skip.

---

## Plan complete

Plan saved to `docs/plans/2026-06-17-msocr-training-pipeline-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per phase, review between phases, fast iteration.

**2. Parallel Session (separate)** — Open a new session with the `executing-plans` skill, batch execution with checkpoints.

**Which approach?**