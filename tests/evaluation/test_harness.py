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