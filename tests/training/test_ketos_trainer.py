"""Tests for msocr.training.ketos_trainer (Kraken 7.0 CLI shape)."""
from unittest.mock import patch, MagicMock
import pytest

from msocr.training.ketos_trainer import KetosTrainer


def _base_config():
    return {
        "dataset": {"format_type": "xml", "base_dir": "/tmp/data"},
        "model": {"spec": "vgg_blocks=4,2,2;conv_size=32,64,128"},
        "training": {
            "epochs": 2,
            "min_epochs": 1,
            "lag": 1,
            "lrate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "Adam",
            "schedule": "constant",
            "augment": True,
            "device": "cpu",
            "precision": "32-true",
            "workers": 1,
            "partition": "train",
        },
        "output": {"model_prefix": "/tmp/out/model"},
    }


def test_train_model_from_scratch_uses_7_0_global_flags(monkeypatch):
    """7.0 moved --device/--workers/--precision to the main `ketos` command, not the `train` subcommand."""
    config = _base_config()
    trainer = KetosTrainer(config)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", fake_run)

    trainer.train_model(["/tmp/data/a.xml"])

    cmd = captured["cmd"]
    # Global flags on the main `ketos` command, before `train`
    assert cmd[0] == "ketos"
    assert "-d" in cmd and cmd[cmd.index("-d") + 1] == "cpu"
    assert "--workers" in cmd
    assert "--precision" in cmd
    # `train` subcommand comes after the global flags
    assert "train" in cmd
    train_idx = cmd.index("train")
    subcmd = cmd[train_idx + 1:]
    # No --device/--workers/--precision on the subcommand
    assert "--device" not in subcmd
    assert "--workers" not in subcmd
    assert "--precision" not in subcmd


def test_train_model_output_is_safetensors(monkeypatch):
    """7.0 default weights format is .safetensors, not .mlmodel."""
    config = _base_config()
    trainer = KetosTrainer(config)
    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", lambda *a, **k: MagicMock(returncode=0))
    # The model prefix is /tmp/out/model; 7.0 produces /tmp/out/model.safetensors
    assert trainer._output_path().suffix == ".safetensors"


def test_train_model_fine_tune_passes_load_resize_freeze_backbone(monkeypatch):
    """Fine-tuning requires --load/--resize/--freeze-backbone on the `train` subcommand."""
    config = _base_config()
    config["training"].update({
        "load": "base.safetensors",
        "resize": "union",
        "freeze_backbone": 5000,
    })
    trainer = KetosTrainer(config)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run", fake_run)
    trainer.train_model("train.arrow", "val.arrow")

    cmd = captured["cmd"]
    assert "--load" in cmd and "base.safetensors" in cmd
    assert "--resize" in cmd and "union" in cmd
    assert "--freeze-backbone" in cmd and "5000" in cmd


def test_get_training_command_dry_run_does_not_execute(monkeypatch):
    """get_training_command returns the command without running it."""
    config = _base_config()
    trainer = KetosTrainer(config)
    called = []
    monkeypatch.setattr("msocr.training.ketos_trainer.subprocess.run",
                        lambda *a, **k: called.append(a) or MagicMock(returncode=0))
    cmd = trainer.get_training_command("train.arrow", "val.arrow")
    assert called == []  # no subprocess.run call
    assert cmd[0] == "ketos"
    assert "train" in cmd


def test_normalization_under_model_section_passes_through():
    """Config puts normalization under model: (about the script), ketos_trainer must still pass --normalization.
    Regression: ketos_trainer.py:94 used to read normalization only from training_cfg, so it was silently dropped.
    """
    config = _base_config()
    config["model"]["normalization"] = "NFD"
    trainer = KetosTrainer(config)
    flags = trainer._train_subcmd_flags("train.arrow", "eval.arrow")
    assert "--normalization" in flags
    assert flags[flags.index("--normalization") + 1] == "NFD"


def test_normalization_under_training_section_passes_through():
    """Docstring schema puts normalization under training:; that path must also work."""
    config = _base_config()
    config["training"]["normalization"] = "NFD"
    trainer = KetosTrainer(config)
    flags = trainer._train_subcmd_flags("train.arrow", "eval.arrow")
    assert "--normalization" in flags
    assert flags[flags.index("--normalization") + 1] == "NFD"