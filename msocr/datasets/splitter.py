"""Dataset splitter for corrected lines."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict


def split_dataset(corrected_dir: Path, train_dir: Path, val_dir: Path, hold_dir: Path, cfg: Dict) -> None:
    xml_files = sorted(corrected_dir.rglob("*.xml"))
    seed = int(cfg.get("random_seed", 42)) if "random_seed" in cfg else 42
    random.seed(seed)
    random.shuffle(xml_files)

    hold_ratio = float(cfg.get("holdout_ratio", 0.05))
    val_ratio = float(cfg.get("validation_ratio", 0.10))

    n = len(xml_files)
    n_hold = max(1, int(n * hold_ratio))
    n_val = max(1, int(n * val_ratio))

    hold = xml_files[:n_hold]
    val = xml_files[n_hold : n_hold + n_val]
    train = xml_files[n_hold + n_val :]

    def copy_split(files, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for xml in files:
            shutil.copy2(xml, dest / xml.name)

    copy_split(train, train_dir)
    copy_split(val, val_dir)
    copy_split(hold, hold_dir)

    manifest = {
        "seed": seed,
        "total": n,
        "train": [f.name for f in train],
        "validation": [f.name for f in val],
        "holdout": [f.name for f in hold],
    }
    (corrected_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
