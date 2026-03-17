"""Payne-Smith dataset split utilities."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict


def _load_region_labels(path: Path | None) -> Dict[str, dict]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_confidence_map(*dirs: Path) -> Dict[str, float]:
    confidence: Dict[str, float] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for json_file in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            line_id = str(payload.get("line_id") or json_file.stem)
            conf = payload.get("bootstrap_confidence")
            if conf is None:
                continue
            confidence[line_id] = float(conf)
    return confidence


def _copy_xml_with_image(xml_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_path, dest_dir / xml_path.name)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        image = xml_path.with_suffix(ext)
        if image.exists():
            shutil.copy2(image, dest_dir / image.name)
            break


def _script_of(xml_path: Path, labels: Dict[str, dict]) -> str:
    info = labels.get(xml_path.stem, {})
    script = str(info.get("script", "Serto")).lower()
    if "estrang" in script or "title" in str(info.get("region_type", "")).lower():
        return "estrangela"
    if "estrang" in xml_path.stem.lower():
        return "estrangela"
    return "serto"


def split_paynesmith_dataset(
    *,
    corrected_dir: Path,
    train_serto_dir: Path,
    train_estrangela_dir: Path,
    validation_serto_dir: Path,
    validation_estrangela_dir: Path,
    holdout_dir: Path,
    manifest_path: Path,
    seed: int = 42,
    train_ratio: float = 0.85,
    validation_ratio: float = 0.10,
    holdout_ratio: float = 0.05,
    random_sample_ratio: float = 0.60,
    uncertainty_sample_ratio: float = 0.40,
    region_labels_path: Path | None = None,
    bootstrap_dir: Path | None = None,
    rejected_dir: Path | None = None,
) -> dict:
    xml_files = sorted(corrected_dir.rglob("*.xml"))
    if not xml_files:
        return {"total": 0, "train": 0, "validation": 0, "holdout": 0}

    rng = random.Random(seed)
    labels = _load_region_labels(region_labels_path)
    confidence = _load_confidence_map(bootstrap_dir or Path(""), rejected_dir or Path(""))

    uncertain_count = int(len(xml_files) * uncertainty_sample_ratio)
    uncertain_sorted = sorted(xml_files, key=lambda p: confidence.get(p.stem, 1.0))
    uncertain_bucket = uncertain_sorted[:uncertain_count]
    remaining = [f for f in xml_files if f not in uncertain_bucket]

    random_count = int(len(xml_files) * random_sample_ratio)
    rng.shuffle(remaining)
    random_bucket = remaining[:random_count]

    selected = list(dict.fromkeys(uncertain_bucket + random_bucket + xml_files))
    rng.shuffle(selected)

    total = len(selected)
    n_holdout = max(1, int(total * holdout_ratio))
    n_validation = max(1, int(total * validation_ratio))
    n_train = max(0, total - n_holdout - n_validation)

    holdout = selected[:n_holdout]
    validation = selected[n_holdout : n_holdout + n_validation]
    train = selected[n_holdout + n_validation : n_holdout + n_validation + n_train]

    train_counts = {"serto": 0, "estrangela": 0}
    validation_counts = {"serto": 0, "estrangela": 0}

    for xml_path in holdout:
        _copy_xml_with_image(xml_path, holdout_dir)

    for xml_path in train:
        script = _script_of(xml_path, labels)
        if script == "estrangela":
            _copy_xml_with_image(xml_path, train_estrangela_dir)
        else:
            _copy_xml_with_image(xml_path, train_serto_dir)
        train_counts[script] += 1

    for xml_path in validation:
        script = _script_of(xml_path, labels)
        if script == "estrangela":
            _copy_xml_with_image(xml_path, validation_estrangela_dir)
        else:
            _copy_xml_with_image(xml_path, validation_serto_dir)
        validation_counts[script] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "total": total,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "holdout": holdout_ratio,
            "random_sample": random_sample_ratio,
            "uncertainty_sample": uncertainty_sample_ratio,
        },
        "counts": {
            "train": train_counts,
            "validation": validation_counts,
            "holdout": len(holdout),
        },
        "train": [p.name for p in train],
        "validation": [p.name for p in validation],
        "holdout": [p.name for p in holdout],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "total": total,
        "train": len(train),
        "validation": len(validation),
        "holdout": len(holdout),
        "train_serto": train_counts["serto"],
        "train_estrangela": train_counts["estrangela"],
    }
