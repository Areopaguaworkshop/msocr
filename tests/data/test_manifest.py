"""Tests for the frozen manifest manager."""

import json
from pathlib import Path

import pytest

from msocr.data.manifest import load_frozen_manifest


def _write_case_files(base_dir: Path, manuscript_id: str, stem: str) -> tuple[Path, Path, Path]:
    case_dir = base_dir / manuscript_id
    case_dir.mkdir(parents=True, exist_ok=True)
    xml_path = case_dir / f"{stem}.xml"
    image_path = case_dir / f"{stem}.png"
    reference_path = case_dir / f"{stem}.txt"
    xml_path.write_text("<PcGts/>", encoding="utf-8")
    image_path.write_bytes(b"png")
    reference_path.write_text("abc", encoding="utf-8")
    return xml_path, image_path, reference_path


def test_load_frozen_manifest_resolves_manifest_id_and_paths(tmp_path: Path):
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    corpus_dir = tmp_path / "corpus"
    _, _, _ = _write_case_files(corpus_dir, "ms001", "line_0001")
    _, holdout_image, holdout_reference = _write_case_files(corpus_dir, "ms002", "line_0002")

    manifest_path = manifests_dir / "printed-v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "printed-v1",
                "language": "syriac",
                "writing_mode": "printed",
                "base_dir": str(corpus_dir),
                "partitions": {
                    "train": [
                        {
                            "id": "line_0001",
                            "xml_path": "ms001/line_0001.xml",
                            "manuscript_id": "ms001",
                        }
                    ],
                    "holdout": [
                        {
                            "id": "line_0002",
                            "image": "ms002/line_0002.png",
                            "reference_text": "ms002/line_0002.txt",
                            "script_variant": "estrangela",
                            "manuscript_id": "ms002",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = load_frozen_manifest("printed-v1", manifests_dir=manifests_dir)

    assert manifest.manifest_id == "printed-v1"
    assert manifest.writing_mode == "printed"
    holdout_case = manifest.get_partition("test")[0]
    assert holdout_case.image == holdout_image
    assert holdout_case.reference_text == holdout_reference
    assert holdout_case.variant == "estrangela"


def test_load_frozen_manifest_rejects_cross_partition_manuscript_overlap(tmp_path: Path):
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    corpus_dir = tmp_path / "corpus"
    _write_case_files(corpus_dir, "ms001", "line_0001")
    _write_case_files(corpus_dir, "ms001", "line_0002")

    manifest_path = manifests_dir / "overlap.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "overlap",
                "base_dir": str(corpus_dir),
                "partitions": {
                    "train": [
                        {
                            "xml_path": "ms001/line_0001.xml",
                            "manuscript_id": "ms001",
                        }
                    ],
                    "holdout": [
                        {
                            "xml_path": "ms001/line_0002.xml",
                            "manuscript_id": "ms001",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manuscript_id"):
        load_frozen_manifest(manifest_path, manifests_dir=manifests_dir)
