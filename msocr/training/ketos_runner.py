"""Ketos training helpers for XML datasets."""

from __future__ import annotations

import subprocess
from typing import Optional


def ketos_train_xml(
    train_glob: str,
    eval_glob: str,
    output_prefix: str,
    device: str = "cuda:0",
    min_epochs: int = 20,
    lag: int = 10,
    augment: bool = True,
) -> None:
    cmd = [
        "ketos",
        "train",
        "-f",
        "xml",
        "--base-dir",
        "R",
        "--device",
        device,
        "--min-epochs",
        str(min_epochs),
        "--lag",
        str(lag),
        "--output",
        output_prefix,
    ]
    if augment:
        cmd.append("--augment")
    cmd.extend([train_glob])
    if eval_glob:
        cmd.extend(["--evaluation-files", eval_glob])
    subprocess.run(cmd, check=True)
