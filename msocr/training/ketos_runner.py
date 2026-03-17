"""Ketos training helpers for XML datasets."""

from __future__ import annotations

import subprocess
from glob import glob
from typing import Sequence


def ketos_train_xml(
    output_prefix: str,
    eval_glob: str = "",
    train_glob: str | None = None,
    train_globs: Sequence[str] | None = None,
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

    sources = list(train_globs or [])
    if train_glob:
        sources.append(train_glob)
    if not sources:
        raise ValueError("At least one training glob is required.")

    expanded: list[str] = []
    for source in sources:
        matches = sorted(glob(source))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(source)

    cmd.extend(expanded)
    if eval_glob:
        eval_matches = sorted(glob(eval_glob))
        cmd.append("--evaluation-files")
        if eval_matches:
            cmd.extend(eval_matches)
        else:
            cmd.append(eval_glob)
    subprocess.run(cmd, check=True)
