"""Dataset splitter for corrected lines."""

from __future__ import annotations

from datetime import datetime
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from msocr.data.manifest import REPO_ROOT


def _load_manuscript_map(raw_value: object, corrected_dir: Path) -> Dict[str, str]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return {str(key): str(value) for key, value in raw_value.items()}

    map_path = Path(str(raw_value))
    if not map_path.is_absolute():
        map_path = (corrected_dir / map_path).resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"Manuscript map not found: {map_path}")
    return json.loads(map_path.read_text(encoding="utf-8"))


def _infer_manuscript_id(
    xml_path: Path,
    corrected_dir: Path,
    manuscript_map: Dict[str, str],
) -> str:
    relative = xml_path.relative_to(corrected_dir)
    lookup_keys = [xml_path.stem, relative.as_posix(), xml_path.name]
    for key in lookup_keys:
        if key in manuscript_map:
            return str(manuscript_map[key])
    if len(relative.parts) > 1:
        return relative.parts[0]
    if "_" in xml_path.stem:
        return xml_path.stem.split("_", 1)[0]
    return xml_path.stem


def _copy_split(files: Iterable[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for xml in files:
        shutil.copy2(xml, dest / xml.name)


def _split_group_ids(
    manuscript_ids: List[str],
    *,
    hold_ratio: float,
    val_ratio: float,
) -> tuple[List[str], List[str], List[str]]:
    total = len(manuscript_ids)
    hold_count = int(round(total * hold_ratio))
    val_count = int(round(total * val_ratio))

    if total >= 3 and hold_ratio > 0:
        hold_count = max(1, hold_count)
    if total >= 2 and val_ratio > 0:
        val_count = max(1, val_count)

    while hold_count + val_count >= total and (hold_count > 0 or val_count > 0):
        if hold_count >= val_count and hold_count > 0:
            hold_count -= 1
        elif val_count > 0:
            val_count -= 1

    hold_ids = manuscript_ids[:hold_count]
    val_ids = manuscript_ids[hold_count : hold_count + val_count]
    train_ids = manuscript_ids[hold_count + val_count :]
    return hold_ids, val_ids, train_ids


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def split_dataset(corrected_dir: Path, train_dir: Path, val_dir: Path, hold_dir: Path, cfg: Dict) -> None:
    xml_files = sorted(corrected_dir.rglob("*.xml"))
    seed = int(cfg.get("random_seed", 42)) if "random_seed" in cfg else 42
    rng = random.Random(seed)

    hold_ratio = float(cfg.get("holdout_ratio", 0.05))
    val_ratio = float(cfg.get("validation_ratio", 0.10))
    manuscript_map = _load_manuscript_map(cfg.get("manuscript_map"), corrected_dir)

    grouped: Dict[str, List[Path]] = {}
    for xml_path in xml_files:
        manuscript_id = _infer_manuscript_id(xml_path, corrected_dir, manuscript_map)
        grouped.setdefault(manuscript_id, []).append(xml_path)

    manuscript_ids = sorted(grouped)
    rng.shuffle(manuscript_ids)
    hold_ids, val_ids, train_ids = _split_group_ids(
        manuscript_ids,
        hold_ratio=hold_ratio,
        val_ratio=val_ratio,
    )

    hold = [xml for manuscript_id in hold_ids for xml in grouped[manuscript_id]]
    val = [xml for manuscript_id in val_ids for xml in grouped[manuscript_id]]
    train = [xml for manuscript_id in train_ids for xml in grouped[manuscript_id]]

    _copy_split(train, train_dir)
    _copy_split(val, val_dir)
    _copy_split(hold, hold_dir)

    manifest_id = str(cfg.get("manifest_id") or f"{corrected_dir.name}-split-v1")

    def partition_rows(files: Iterable[Path]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for xml_path in sorted(files):
            rows.append(
                {
                    "id": xml_path.stem,
                    "xml_path": _path_for_manifest(xml_path),
                    "manuscript_id": _infer_manuscript_id(
                        xml_path, corrected_dir, manuscript_map
                    ),
                }
            )
        return rows

    manifest = {
        "manifest_id": manifest_id,
        "schema_version": "msocr.manifest.v1",
        "kind": "split",
        "generated_at": datetime.now().isoformat(),
        "writing_mode": str(cfg.get("writing_mode", "printed")).lower(),
        "language": cfg.get("language"),
        "dvc_tracked": bool(cfg.get("dvc_tracked", False)),
        "seed": seed,
        "total": len(xml_files),
        "split_policy": {
            "unit": "manuscript_id",
            "holdout_ratio": hold_ratio,
            "validation_ratio": val_ratio,
        },
        "manuscripts": {
            "train": sorted(train_ids),
            "validation": sorted(val_ids),
            "holdout": sorted(hold_ids),
        },
        "partitions": {
            "train": partition_rows(train),
            "validation": partition_rows(val),
            "holdout": partition_rows(hold),
        },
    }
    (corrected_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
