"""Tests for msocr.training.orchestrator. Runner and eval are mocked."""
import json
from unittest.mock import patch, MagicMock

from msocr.training.orchestrator import walk_style_group


def _make_manifest(tmp_path, style_groups=None):
    """Build a tiny fixture manifest on disk with 1 train + 1 val + 1 holdout in g1."""
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
        "style_groups": style_groups or {"g1": {"manuscript_ids": ["M1", "M2", "M3"]}},
    }
    p = tmp_path / "test-v1.json"
    p.write_text(json.dumps(manifest))
    return p


def test_walk_style_group_builds_train_cmd_and_runs_eval(tmp_path):
    """walk_style_group builds the 7.0 ketos train command, uploads .arrow,
    runs training, then runs eval on the downloaded model."""
    manifest_path = _make_manifest(tmp_path)

    fake_runner = MagicMock()
    # run_training returns the artifact_local_path it was given.
    fake_runner.run_training.return_value = "/tmp/out.safetensors"

    fake_arrow_paths = {
        "a": str(tmp_path / "train.arrow"),
        "b": str(tmp_path / "val.arrow"),
    }

    def fake_compile(self, xmls):
        # train partition has "a.xml", val partition has "b.xml"
        stem = __import__("pathlib").Path(xmls[0]).stem
        return fake_arrow_paths.get(stem, str(tmp_path / "x.arrow"))

    fake_report = {"per_style_group": {"g1": {"cer": 0.05}}, "per_manuscript": {}}

    with patch("msocr.training.orchestrator.KetosTrainer.compile_dataset",
               fake_compile), \
         patch("msocr.training.orchestrator.run_evaluation",
               return_value=fake_report) as fake_eval:
        report = walk_style_group(
            manifest_path=str(manifest_path),
            style_group_id="g1",
            runner=fake_runner,
            base_model_path="/tmp/base.safetensors",
            output_model_path="/tmp/out.safetensors",
            reports_dir="/tmp/reports",
        )

    # run_training called once with the 7.0 ketos train command.
    fake_runner.run_training.assert_called_once()
    _, kwargs = fake_runner.run_training.call_args
    train_cmd = kwargs["train_cmd"]
    assert train_cmd[:7] == [
        "ketos", "-d", "cuda:0", "--workers", "8", "train",
        "--load", "/tmp/base.safetensors",
    ][:7]
    assert "--resize" in train_cmd
    assert "--augment" in train_cmd
    # ponytail: --augment must appear exactly once (plan had a dup bug).
    assert train_cmd.count("--augment") == 1
    # pre_train_upload carries the compiled .arrow files for the pod.
    assert "pre_train_upload" in kwargs
    uploads = kwargs["pre_train_upload"]
    assert uploads == [
        (str(tmp_path / "train.arrow"), "/workspace/train.arrow"),
        (str(tmp_path / "val.arrow"), "/workspace/val.arrow"),
    ]

    # run_evaluation called once with the downloaded model path + style_group.
    fake_eval.assert_called_once()
    _, eval_kwargs = fake_eval.call_args
    assert eval_kwargs["model_path"] == "/tmp/out.safetensors"
    assert eval_kwargs["style_group_id"] == "g1"
    assert report == fake_report