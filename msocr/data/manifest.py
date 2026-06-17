"""Frozen manifest manager for reproducible Sogdian HTR training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from msocr.language_registry import VALID_SCRIPT_BLOCKS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"

_PARTITION_ALIASES = {
    "val": "validation",
    "dev": "validation",
    "test": "holdout",
    "eval": "holdout",
}


@dataclass(frozen=True)
class ManifestCase:
    """Single immutable manifest entry."""

    id: str
    manuscript_id: str
    language: Optional[str] = None
    image: Optional[Path] = None
    reference_text: Optional[Path] = None
    xml_path: Optional[Path] = None
    engine: str = "auto"
    model: Optional[str] = None
    variant: str = "default"
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenManifest:
    """Resolved manifest with normalized partitions and file paths."""

    manifest_id: str
    path: Path
    writing_mode: str
    language: Optional[str]
    dvc_tracked: bool
    metadata: Dict[str, Any]
    partitions: Dict[str, List[ManifestCase]]
    script_block: str = ""
    style_groups: Optional[Dict[str, Dict[str, Any]]] = None

    def get_partition(self, name: str) -> List[ManifestCase]:
        partition_name = _canonical_partition_name(name)
        if partition_name in self.partitions:
            return list(self.partitions[partition_name])
        if "cases" in self.partitions and partition_name in {
            "cases",
            "train",
            "validation",
            "holdout",
        }:
            return list(self.partitions["cases"])
        raise KeyError(f"Partition not found in manifest {self.manifest_id}: {name}")


def _canonical_partition_name(name: str) -> str:
    key = name.strip().lower()
    return _PARTITION_ALIASES.get(key, key)


def _resolve_manifest_path(reference: str | Path, manifests_dir: Path) -> Path:
    candidate = Path(reference)
    if candidate.exists():
        return candidate.resolve()

    manifest_key = str(reference).strip()
    for suffix in (".json", ".jsonl"):
        manifest_path = manifests_dir / f"{manifest_key}{suffix}"
        if manifest_path.exists():
            return manifest_path.resolve()

    raise FileNotFoundError(
        f"Manifest not found: {reference}. Looked in {manifests_dir}."
    )


def _read_manifest_payload(manifest_path: Path) -> Any:
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return {"cases": []}

    if manifest_path.suffix.lower() == ".jsonl":
        return {"cases": [json.loads(line) for line in text.splitlines() if line.strip()]}

    return json.loads(text)


def _resolve_base_dir(raw_base_dir: Any, manifest_dir: Path) -> Optional[Path]:
    if not raw_base_dir:
        return None

    base_dir = Path(str(raw_base_dir))
    if base_dir.is_absolute():
        return base_dir.resolve()

    repo_candidate = (REPO_ROOT / base_dir).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return (manifest_dir / base_dir).resolve()


def _resolve_optional_path(
    raw_value: Any,
    *,
    base_dir: Optional[Path],
    manifest_dir: Path,
) -> Optional[Path]:
    if raw_value in (None, ""):
        return None

    path = Path(str(raw_value))
    if path.is_absolute():
        return path.resolve()

    candidates = []
    if base_dir is not None:
        candidates.append((base_dir / path).resolve())
    candidates.append((manifest_dir / path).resolve())
    candidates.append((REPO_ROOT / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _infer_manuscript_id(
    *,
    xml_path: Optional[Path],
    image_path: Optional[Path],
    reference_text_path: Optional[Path],
    base_dir: Optional[Path],
    manifest_dir: Path,
) -> str:
    for path in (xml_path, image_path, reference_text_path):
        if path is None:
            continue
        for anchor in (base_dir, manifest_dir):
            if anchor is None:
                continue
            try:
                relative = path.relative_to(anchor)
            except ValueError:
                continue
            if len(relative.parts) > 1:
                return relative.parts[0]
        if path.parent.name:
            return path.parent.name
        return path.stem
    return "unknown"


def _default_case_id(
    *,
    xml_path: Optional[Path],
    image_path: Optional[Path],
    reference_text_path: Optional[Path],
    partition_name: str,
    index: int,
) -> str:
    for path in (xml_path, image_path, reference_text_path):
        if path is not None:
            return path.stem
    return f"{partition_name}_{index:04d}"


def _coerce_case_item(
    item: Any,
    *,
    partition_name: str,
    index: int,
    base_dir: Optional[Path],
    manifest_dir: Path,
    defaults: Dict[str, Any],
) -> ManifestCase:
    if isinstance(item, (str, Path)):
        item = {"xml_path": str(item)}
    if not isinstance(item, dict):
        raise ValueError(
            f"Manifest partition {partition_name!r} contains unsupported item: {item!r}"
        )

    xml_path = _resolve_optional_path(
        item.get("xml_path") or item.get("xml"),
        base_dir=base_dir,
        manifest_dir=manifest_dir,
    )
    image_path = _resolve_optional_path(
        item.get("image"),
        base_dir=base_dir,
        manifest_dir=manifest_dir,
    )
    reference_text_path = _resolve_optional_path(
        item.get("reference_text"),
        base_dir=base_dir,
        manifest_dir=manifest_dir,
    )

    case_id = str(
        item.get("id")
        or _default_case_id(
            xml_path=xml_path,
            image_path=image_path,
            reference_text_path=reference_text_path,
            partition_name=partition_name,
            index=index,
        )
    )
    language = item.get("language", defaults.get("language"))
    if language is not None:
        language = str(language).strip().lower()
    variant = str(
        item.get(
            "variant",
            item.get("script_variant", defaults.get("script_variant", "default")),
        )
    ).strip().lower()
    manuscript_id = str(
        item.get("manuscript_id")
        or _infer_manuscript_id(
            xml_path=xml_path,
            image_path=image_path,
            reference_text_path=reference_text_path,
            base_dir=base_dir,
            manifest_dir=manifest_dir,
        )
    )

    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "id",
            "language",
            "image",
            "reference_text",
            "xml_path",
            "xml",
            "manuscript_id",
            "engine",
            "model",
            "variant",
            "script_variant",
            "device",
        }
    }

    return ManifestCase(
        id=case_id,
        manuscript_id=manuscript_id,
        language=language,
        image=image_path,
        reference_text=reference_text_path,
        xml_path=xml_path,
        engine=str(item.get("engine", defaults.get("engine", "auto"))).lower(),
        model=item.get("model", defaults.get("model")),
        variant=variant,
        device=str(item.get("device", defaults.get("device", "cpu"))),
        metadata=metadata,
    )


def _extract_partition_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "partitions" in payload and isinstance(payload["partitions"], dict):
        return payload["partitions"]
    if "splits" in payload and isinstance(payload["splits"], dict):
        return payload["splits"]

    legacy_partitions = {
        key: payload[key]
        for key in ("train", "validation", "holdout", "test", "cases")
        if key in payload
    }
    if legacy_partitions:
        return legacy_partitions

    if "cases" in payload and isinstance(payload["cases"], list):
        return {"cases": payload["cases"]}

    raise ValueError(
        "Unsupported manifest format. Expected partitions/splits or a cases list."
    )


def _validate_manuscript_overlap(partitions: Dict[str, List[ManifestCase]]) -> None:
    owners: Dict[str, str] = {}
    for partition_name in ("train", "validation", "holdout"):
        for case in partitions.get(partition_name, []):
            manuscript_id = case.manuscript_id.strip()
            if not manuscript_id or manuscript_id == "unknown":
                continue
            existing = owners.get(manuscript_id)
            if existing and existing != partition_name:
                raise ValueError(
                    "Frozen manifest must keep manuscript_id isolated by partition. "
                    f"Found {manuscript_id!r} in both {existing!r} and {partition_name!r}."
                )
            owners[manuscript_id] = partition_name


def load_frozen_manifest(
    reference: str | Path,
    *,
    manifests_dir: Optional[Path] = None,
) -> FrozenManifest:
    manifests_root = (manifests_dir or DEFAULT_MANIFESTS_DIR).resolve()
    manifest_path = _resolve_manifest_path(reference, manifests_root)
    payload = _read_manifest_payload(manifest_path)

    if isinstance(payload, list):
        payload = {"cases": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported manifest payload in {manifest_path}")

    base_dir = _resolve_base_dir(payload.get("base_dir"), manifest_path.parent)
    defaults = {
        "language": payload.get("language"),
        "script_variant": payload.get("script_variant"),
        "engine": payload.get("engine", "auto"),
        "model": payload.get("model"),
        "device": payload.get("device", "cpu"),
    }

    raw_partitions = _extract_partition_payload(payload)
    partitions: Dict[str, List[ManifestCase]] = {}
    for raw_name, raw_items in raw_partitions.items():
        partition_name = _canonical_partition_name(raw_name)
        if not isinstance(raw_items, list):
            raise ValueError(
                f"Manifest partition {raw_name!r} must be a list, got {type(raw_items)}"
            )
        partitions[partition_name] = [
            _coerce_case_item(
                item,
                partition_name=partition_name,
                index=index,
                base_dir=base_dir,
                manifest_dir=manifest_path.parent,
                defaults=defaults,
            )
            for index, item in enumerate(raw_items, start=1)
        ]

    _validate_manuscript_overlap(partitions)

    manifest_id = str(payload.get("manifest_id") or payload.get("id") or manifest_path.stem)
    writing_mode = str(payload.get("writing_mode", "handwritten")).strip().lower()
    language = payload.get("language")
    if language is not None:
        language = str(language).strip().lower()

    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "manifest_id",
            "id",
            "writing_mode",
            "language",
            "partitions",
            "splits",
            "cases",
            "train",
            "validation",
            "holdout",
            "test",
            "base_dir",
            "dvc_tracked",
            "script_block",
            "style_groups",
        }
    }

    script_block = payload.get("script_block")
    if not script_block:
        raise ValueError(
            f"Manifest {manifest_path} missing required 'script_block' field"
        )
    if script_block not in VALID_SCRIPT_BLOCKS:
        raise ValueError(
            f"Manifest {manifest_path} has invalid script_block {script_block!r}; "
            f"valid: {sorted(VALID_SCRIPT_BLOCKS)}"
        )

    return FrozenManifest(
        manifest_id=manifest_id,
        path=manifest_path,
        writing_mode=writing_mode,
        language=language,
        dvc_tracked=bool(
            payload.get("dvc_tracked", False)
            or manifest_path.with_suffix(f"{manifest_path.suffix}.dvc").exists()
        ),
        metadata=metadata,
        partitions=partitions,
        script_block=script_block,
        style_groups=payload.get("style_groups"),
    )


def iter_partition_cases(
    manifest: FrozenManifest,
    partition_names: Iterable[str],
) -> List[ManifestCase]:
    """Return the first non-empty partition from the provided names."""

    for name in partition_names:
        try:
            cases = manifest.get_partition(name)
        except KeyError:
            continue
        if cases:
            return cases
    return []


def iter_style_group_cases(
    manifest: FrozenManifest,
    style_group_id: str,
    partition: str = "train",
) -> List[ManifestCase]:
    """Yield ManifestCase objects for a style_group's manuscripts in a partition.

    Raises ValueError if the style_group_id is not in the manifest.
    """
    if not manifest.style_groups or style_group_id not in manifest.style_groups:
        raise ValueError(
            f"style_group {style_group_id!r} not in manifest {manifest.manifest_id}"
        )
    ms_ids = set(manifest.style_groups[style_group_id].get("manuscript_ids", []))
    return [
        case for case in manifest.get_partition(partition)
        if case.manuscript_id in ms_ids
    ]
