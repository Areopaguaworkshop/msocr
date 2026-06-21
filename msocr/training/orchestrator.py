"""Procedural per-style-group training orchestrator.

Per design D8 (Approach A): walk a style_group in a manifest,
compile train+val .arrow locally, upload them to a RunPod pod,
run training, download the .safetensors artifact, run evaluation
locally on the downloaded model. One style-group at a time.
No queue, no DAG.

Ponytail: if we need durable parallelism later, wrap this in RQ.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from msocr.data.manifest import load_frozen_manifest, iter_style_group_cases
from msocr.training.ketos_trainer import KetosTrainer
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
    load_model_path = Path(load_model)
    if not load_model_path.exists():
        raise FileNotFoundError(f"Base model for RunPod training not found: {load_model_path}")

    # Compile train + val .arrow locally, then upload to the pod (Task 7.3).
    train_cases = iter_style_group_cases(manifest, style_group_id, partition="train")
    val_cases = iter_style_group_cases(manifest, style_group_id, partition="validation")
    train_xmls = [str(c.xml_path) for c in train_cases if c.xml_path]
    val_xmls = [str(c.xml_path) for c in val_cases if c.xml_path]
    # ponytail: one KetosTrainer per partition with its own output prefix in a
    # tmp dir; the .arrow files are uploaded, the tmp dir is throwaway.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_cfg = _dataset_cfg(tmp_path / "train")
        val_cfg = _dataset_cfg(tmp_path / "val")
        train_arrow = KetosTrainer(train_cfg).compile_dataset(train_xmls)
        val_arrow = KetosTrainer(val_cfg).compile_dataset(val_xmls)
        pre_train_upload = [
            (train_arrow, "/workspace/train.arrow"),
            (val_arrow, "/workspace/val.arrow"),
            (str(load_model_path), "/workspace/base.safetensors"),
        ]

        # Build the ketos train command (7.0 global flags).
        # ponytail: --augment is in the list once; the plan had a dup `if augment`
        # append that would add it twice — fixed.
        train_cmd = [
            "ketos", "-d", device, "--workers", str(workers), "train",
            "--load", "/workspace/base.safetensors",
            "--resize", "union",
            "--freeze-backbone", str(freeze_backbone),
            "--epochs", str(epochs),
            "--min-epochs", str(min_epochs),
            "--lag", str(lag),
            "-f", "binary",
            "-t", "/workspace/train.arrow",
            "-e", "/workspace/val.arrow",
            "-o", "/workspace/models/" + style_group_id,
        ]
        if augment:
            train_cmd.append("--augment")

        runner.run_training(
            name=f"{manifest.manifest_id}-{style_group_id}",
            train_cmd=train_cmd,
            artifact_remote_path=f"/workspace/models/{style_group_id}.safetensors",
            artifact_local_path=output_model_path,
            pre_train_upload=pre_train_upload,
        )

    return run_evaluation(
        manifest_path=manifest_path,
        style_group_id=style_group_id,
        model_path=output_model_path,
        reports_dir=reports_dir,
    )


def _dataset_cfg(prefix: Path) -> dict:
    """Minimal KetosTrainer config for a one-shot compile_dataset call."""
    return {
        "dataset": {"format_type": "xml"},
        "model": {"spec": "placeholder"},
        "training": {"epochs": 0, "device": "cpu", "workers": 1},
        "output": {"model_prefix": str(prefix.with_suffix(""))},
    }
