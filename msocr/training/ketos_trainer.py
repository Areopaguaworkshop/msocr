"""Kraken 7.0 ketos CLI wrapper.

7.0 changes (https://github.com/mittagessen/kraken/releases/tag/7.0):
- --device/--workers/--precision/--threads are global on `ketos`, not on `train`.
- Default output is .safetensors (was .mlmodel).
- --training-files/--evaluation-files renamed to --training-data/--evaluation-data.
- --load/--resize/--freeze-backbone for fine-tuning.
- YAML experiment configs supported via `ketos train --config experiment.yaml`.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class KetosTrainer:
    """Wraps the Kraken 7.0 `ketos` CLI for compile/train/test.

    Config sections (mirrors the YAML experiment schema):
      dataset: {format_type: xml|alto|page|path|binary, base_dir: str}
      model:   {spec: str}
      training: {epochs, min_epochs, lag, lrate, weight_decay, optimizer,
                 schedule, augment, warmup, batch_size, device, precision,
                 workers, partition, load, resize, freeze_backbone, normalization}
      output:  {model_prefix: str}
    """

    config: dict

    def __post_init__(self) -> None:
        self.dataset_cfg = self.config.get("dataset", {})
        self.model_cfg = self.config.get("model", {})
        self.training_cfg = self.config.get("training", {})
        self.output_cfg = self.config.get("output", {})

    # ponytail: no validate-on-init; validate_config() is called explicitly by train().
    # Adding a schema lib here would be overkill for a 4-section dict.

    def _global_flags(self) -> list[str]:
        """7.0 global flags on the main `ketos` command."""
        flags = []
        device = self.training_cfg.get("device", "auto")
        flags += ["-d", device]
        workers = str(self.training_cfg.get("workers", 1))
        flags += ["--workers", workers]
        precision = self.training_cfg.get("precision", "32-true")
        flags += ["--precision", precision]
        return flags

    def _train_subcmd_flags(self, train_data: str, eval_data: str | None) -> list[str]:
        """7.0 `train` subcommand flags."""
        t = self.training_cfg
        flags = ["train"]
        flags += ["-t", train_data]
        if eval_data:
            flags += ["-e", eval_data]
        fmt = self.dataset_cfg.get("format_type", "xml")
        flags += ["-f", fmt]
        flags += ["-o", self._output_prefix()]
        # Fine-tuning flags (7.0)
        load = t.get("load")
        if load:
            flags += ["--load", str(load)]
        resize = t.get("resize")
        if resize:
            flags += ["--resize", resize]
        freeze = t.get("freeze_backbone")
        if freeze is not None:
            flags += ["--freeze-backbone", str(freeze)]
        # Training hyperparams
        if "epochs" in t:
            flags += ["--epochs", str(t["epochs"])]
        if "min_epochs" in t:
            flags += ["--min-epochs", str(t["min_epochs"])]
        if "lag" in t:
            flags += ["--lag", str(t["lag"])]
        if "lrate" in t:
            flags += ["--lrate", str(t["lrate"])]
        if "weight_decay" in t:
            flags += ["--weight-decay", str(t["weight_decay"])]
        if "optimizer" in t:
            flags += ["--optimizer", t["optimizer"]]
        if "schedule" in t:
            flags += ["--schedule", t["schedule"]]
        if "warmup" in t:
            flags += ["--warmup", str(t["warmup"])]
        if "batch_size" in t:
            flags += ["-B", str(t["batch_size"])]
        if t.get("augment"):
            flags += ["--augment"]
        # ponytail: normalization may live under training: (per docstring) or model: (where
        # the config author put it, semantically about the script). Check both so either works.
        norm = t.get("normalization") or self.model_cfg.get("normalization")
        if norm:
            flags += ["--normalization", norm]
        return flags

    def _output_prefix(self) -> str:
        return str(self.output_cfg["model_prefix"])

    def _output_path(self) -> Path:
        """7.0 default output is .safetensors."""
        return Path(self._output_prefix()).with_suffix(".safetensors")

    def compile_dataset(self, xml_files: Sequence[str]) -> str:
        """ketos compile -f xml -o <out> <xml_files...>"""
        out = self._output_prefix() + ".arrow"
        cmd = ["ketos", "compile", "-f", self.dataset_cfg.get("format_type", "xml"),
               "-o", out, *xml_files]
        subprocess.run(cmd, check=True)
        return out

    def train_model(self, train_data: str, eval_data: str | None = None) -> Path:
        cmd = ["ketos", *self._global_flags(), *self._train_subcmd_flags(train_data, eval_data)]
        subprocess.run(cmd, check=True)
        return self._output_path()

    def test_model(self, model_path: str | Path, test_data: str) -> str:
        """ketos test -m <model> -f <fmt> <test_data> -> returns stdout (CER/WER JSON)."""
        cmd = ["ketos", *self._global_flags(), "test",
               "-m", str(model_path),
               "-f", self.dataset_cfg.get("format_type", "binary"),
               test_data]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout

    def validate_config(self) -> None:
        required = [("model.spec", self.model_cfg.get("spec")),
                    ("dataset.format_type", self.dataset_cfg.get("format_type")),
                    ("training.epochs", self.training_cfg.get("epochs")),
                    ("output.model_prefix", self.output_cfg.get("model_prefix"))]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"ketos_trainer config missing: {', '.join(missing)}")

    def train(self, xml_files: Sequence[str], eval_xml_files: Sequence[str] | None = None) -> Path:
        """Full pipeline: validate -> compile -> train. Returns output model path."""
        self.validate_config()
        train_arrow = self.compile_dataset(xml_files)
        eval_arrow = self.compile_dataset(list(eval_xml_files)) if eval_xml_files else None
        # ponytail: compile eval into a separate .arrow; could share but separate is clearer.
        return self.train_model(train_arrow, eval_arrow)

    def get_training_command(self, train_data: str, eval_data: str | None = None) -> list[str]:
        """Dry-run: returns the command that train_model would run, without executing."""
        return ["ketos", *self._global_flags(), *self._train_subcmd_flags(train_data, eval_data)]