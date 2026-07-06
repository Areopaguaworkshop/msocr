"""Phase 0b - Python-API Kraken training harness with per-row gradient freeze.

This is the §1 mechanism from `docs/fixa-v4-impl-plan-2026-07-06.md`. The
existing `KetosTrainer` (`msocr/training/ketos_trainer.py`) shells out to the
`ketos` CLI, which has no per-row classifier-freeze flag. To freeze only the
37 old-class rows (leaving the 5 new Sogdian rows trainable) we must drive
`KrakenTrainer` (Lightning) from Python so a `tensor.register_hook` can zero
the old-row gradients before each optimizer step.

Verified API (kraken 7.0.2, installed):
  from kraken.train import KrakenTrainer, VGSLRecognitionModel, VGSLRecognitionDataModule
  from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig
  m_config = VGSLRecognitionTrainingConfig(**params)   # flat kwargs from CLI
  dm_config = VGSLRecognitionTrainingDataConfig(**params)
  dm = VGSLRecognitionDataModule(dm_config)
  trainer = KrakenTrainer(...)
  with trainer.init_module(empty_init=False):
      model = VGSLRecognitionModel.load_from_weights(load_path, config=m_config)
  trainer.fit(model, dm)
The CLI then converts the best checkpoint to `best_{score:.4f}.safetensors`
via `kraken.models.convert.convert_models`.

Why per-row freeze is sound here (see docs/fixa-v4-impl-plan-2026-07-06.md §0):
  - `freeze_backbone=999999` keeps CNN/LSTM frozen the whole run, so old-row
    weights are constant once their grads are zeroed.
  - Softmax renormalization shifts old-class probabilities as new rows learn
    - the desired signal, not drift.
  - `register_hook` on a leaf `nn.Parameter` persists for the tensor's
    lifetime (PyTorch source), so one registration in `on_train_start` fires
    every optimizer step.

Public entrypoint: `train_with_row_freeze(cfg) -> Path`. Returns the path to
the best `*.safetensors` checkpoint, matching what the CLI would have
written so downstream eval (`triangulate_eval.py`, `confusion_audit.py`,
`harness.py`) is unchanged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig
from kraken.models.convert import convert_models
from kraken.train import KrakenTrainer, VGSLRecognitionDataModule, VGSLRecognitionModel
from kraken.train.utils import KrakenOnExceptionCheckpoint

REPORTS_DIR = Path("reports")
DEFAULT_CODEC_JSON = REPORTS_DIR / "c2av_union_codec.json"


@dataclass
class RowFreezeConfig:
    """Mirrors the v3 shipped recipe; fields map 1:1 to ketos CLI kwargs."""

    # Model / data
    load_model: str
    train_xml_files: Sequence[str]
    eval_xml_files: Sequence[str] | None = None
    format_type: str = "page"
    resize: str = "union"
    checkpoint_path: str = "c2av_finetune_v4_freeze"
    weights_format: str = "safetensors"

    # Training hyperparams (v3 shipped recipe defaults)
    epochs: int = 200
    min_epochs: int = 8
    lag: int = 25
    lrate: float = 1e-4
    warmup: int = 200
    quit: str = "early"
    freeze_backbone: int = 999999  # samples to keep backbone frozen (whole run)
    augment: bool = True
    batch_size: int = 16
    optimizer: str = "AdamW"
    schedule: str = "cosine"
    weight_decay: float = 0.0
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: str = "auto"
    num_workers: int = 1
    num_threads: int = 1
    reorder: str = "rtl"  # Sogdian RTL (AGENTS.md convention)
    bidi_reordering: bool = True
    normalize_whitespace: bool = True
    legacy_polygons: bool = False
    padding: int = 16
    freq: float = 1.0
    deterministic: bool = False

    # Row-freeze specifics
    freeze_old_rows: bool = True
    codec_json: str = str(DEFAULT_CODEC_JSON)


def _load_row_map(codec_json: Path) -> tuple[list[int], list[int]]:
    """Return (old_rows, new_rows) from the Phase 0a dump."""
    data = json.loads(codec_json.read_text())
    return list(data["old_rows"]), list(data["new_rows"])


def _register_freeze_hook(param: torch.nn.Parameter, old_rows: list[int]) -> None:
    """Zero the gradient for old-class rows on every backward pass.

    `register_hook` on a leaf Parameter persists across optimizer steps.
    """
    n_rows = param.shape[0]
    mask = torch.zeros(n_rows, dtype=torch.bool, device=param.device)
    for r in old_rows:
        if 0 <= r < n_rows:
            mask[r] = True
    keep = (~mask).to(param.dtype)

    def _zero_old_rows(grad: torch.Tensor) -> torch.Tensor:
        # ponytail: handle 1D (bias) and 2D (weight) grads. keep.unsqueeze(1)
        # on a 1D grad broadcasts [n_rows] x [n_rows,1] -> [n_rows,n_rows] and
        # PyTorch rejects the size change. View keep to match grad's leading dim.
        shape = [n_rows] + [1] * (grad.dim() - 1)
        return grad * keep.view(*shape)

    param.register_hook(_zero_old_rows)


class RowFreezeRecognitionModel(VGSLRecognitionModel):
    """VGSLRecognitionModel that freezes old-class classifier rows after setup.

    Lightning calls `setup(stage='fit')` once before training; the base
    `setup()` loads the model, resizes the output via `--resize union`, and
    encodes the dataset. We register the freeze hooks in `on_train_start`,
    which fires after `setup()` and before the first backward pass - the
    resized 42-row classifier exists by then.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ponytail: set after construction via `enable_row_freeze()` because
        # `load_from_weights(path, config)` takes only (path, config) and
        # `load_from_checkpoint` forwards kwargs to Lightning hparams, not __init__.
        self._freeze_old_rows = False
        self._codec_json = DEFAULT_CODEC_JSON
        self._hooks_registered = False

    def enable_row_freeze(self, codec_json: str | Path = DEFAULT_CODEC_JSON) -> None:
        """Opt into the §1 row freeze. Call after load_from_weights/_checkpoint."""
        self._freeze_old_rows = True
        self._codec_json = Path(codec_json)

    def on_train_start(self) -> None:  # type: ignore[override]
        super().on_train_start()
        if not self._freeze_old_rows or self._hooks_registered:
            return
        old_rows, _new_rows = _load_row_map(self._codec_json)
        net = self.net
        if net is None:
            raise RuntimeError("Model has no network (self.net is None) at on_train_start.")
        # ponytail: net.nn is MultiParamSequential; nn[-1] is LinSoftmax for
        # recognition models (verified on sophro_mhiro_syriac.mlmodel).
        from kraken.lib.vgsl.layers import LinSoftmax  # type: ignore[import]
        if not isinstance(net.nn[-1], LinSoftmax):
            raise RuntimeError(
                f"Expected LinSoftmax as last layer, got {type(net.nn[-1]).__name__}. "
                "Row freeze only applies to recognition (CTC) models."
            )
        lin = net.nn[-1].lin
        n_rows = lin.weight.shape[0]
        if max(old_rows) >= n_rows:
            raise RuntimeError(
                f"Old-row index {max(old_rows)} >= classifier rows {n_rows}. "
                f"Codec JSON `{self._codec_json}` is stale vs the loaded model."
            )
        _register_freeze_hook(lin.weight, old_rows)
        _register_freeze_hook(lin.bias, old_rows)
        self._hooks_registered = True
        print(f"[row-freeze] registered gradient hooks on {len(old_rows)} old "
              f"rows (0..{max(old_rows)}); leaving {n_rows - len(old_rows)} new "
              f"rows trainable.", file=sys.stderr)


def _build_params_dict(cfg: RowFreezeConfig) -> dict[str, Any]:
    """Flat kwargs dict accepted by both VGSLRecognitionTrainingConfig and
    VGSLRecognitionTrainingDataConfig (they pop what they need and pass the
    rest up to TrainingConfig / Config)."""
    return {
        "training_data": list(cfg.train_xml_files),
        "evaluation_data": list(cfg.eval_xml_files) if cfg.eval_xml_files else None,
        "format_type": cfg.format_type,
        "resize": cfg.resize,
        "checkpoint_path": cfg.checkpoint_path,
        "weights_format": cfg.weights_format,
        "epochs": cfg.epochs,
        "min_epochs": cfg.min_epochs,
        "lag": cfg.lag,
        "lrate": cfg.lrate,
        "warmup": cfg.warmup,
        "quit": cfg.quit,
        "freeze_backbone": cfg.freeze_backbone,
        "augment": cfg.augment,
        "batch_size": cfg.batch_size,
        "optimizer": cfg.optimizer,
        "schedule": cfg.schedule,
        "weight_decay": cfg.weight_decay,
        "gradient_clip_val": cfg.gradient_clip_val,
        "accumulate_grad_batches": cfg.accumulate_grad_batches,
        "precision": cfg.precision,
        "accelerator": cfg.accelerator,
        "device": cfg.devices,
        "num_workers": cfg.num_workers,
        "num_threads": cfg.num_threads,
        "reorder": cfg.reorder,
        "bidi_reordering": cfg.bidi_reordering,
        "normalize_whitespace": cfg.normalize_whitespace,
        "legacy_polygons": cfg.legacy_polygons,
        "padding": cfg.padding,
        "freq": cfg.freq,
        "partition": 1 if cfg.eval_xml_files else 0.9,
    }


def train_with_row_freeze(cfg: RowFreezeConfig) -> Path:
    """Run training with the §1 classifier-row freeze. Returns best checkpoint path.

    Emits `best_{score:.4f}.safetensors` to the `checkpoint_path` dir, matching
    what the `ketos train` CLI writes, so downstream eval scripts work unchanged.
    """
    params = _build_params_dict(cfg)

    dm_config = VGSLRecognitionTrainingDataConfig(**params)
    m_config = VGSLRecognitionTrainingConfig(**params)

    data_module = VGSLRecognitionDataModule(dm_config)

    # Match the CLI's callback wiring exactly.
    cbs: list[Any] = [
        KrakenOnExceptionCheckpoint(dirpath=Path(cfg.checkpoint_path),
                                    filename="checkpoint_abort"),
        ModelCheckpoint(dirpath=Path(cfg.checkpoint_path),
                        save_top_k=10,
                        monitor="val_metric",
                        mode="max",
                        auto_insert_metric_name=False,
                        filename="checkpoint_{epoch:02d}-{val_metric:.4f}"),
    ]

    val_check_interval: dict[str, Any] = (
        {"check_val_every_n_epoch": int(cfg.freq)} if cfg.freq > 1
        else {"val_check_interval": cfg.freq}
    )

    trainer = KrakenTrainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        max_epochs=cfg.epochs if cfg.quit == "fixed" else -1,
        min_epochs=cfg.min_epochs,
        enable_progress_bar=True,
        deterministic=cfg.deterministic,
        enable_model_summary=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=cbs,
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
        **val_check_interval,
    )

    with trainer.init_module(empty_init=False):
        load_path = cfg.load_model
        if load_path.endswith("ckpt"):
            model = RowFreezeRecognitionModel.load_from_checkpoint(
                load_path, config=m_config, weights_only=False,
            )
        else:
            model = RowFreezeRecognitionModel.load_from_weights(
                load_path, config=m_config,
            )
        if cfg.freeze_old_rows:
            model.enable_row_freeze(cfg.codec_json)

    trainer.fit(model, data_module)

    # Find the best checkpoint and convert to safetensors, mirroring the CLI.
    ckpt_cb = next(c for c in cbs if isinstance(c, ModelCheckpoint))
    score = ckpt_cb.best_model_score
    score_val = score.item() if score is not None else 0.0
    weight_path = Path(ckpt_cb.best_model_path).with_name(
        f"best_{score_val:.4f}.{cfg.weights_format}"
    )
    opath = convert_models([ckpt_cb.best_model_path], weight_path,
                           weights_format=cfg.weights_format)
    return Path(opath)


# ponytail: one-runnable self-check so the harness doesn't silently break.
# Verifies hook registration + mask shape without running a real training.
def _self_check() -> None:
    """Verify the freeze hook zeros old-row grads on a tiny tensor."""
    w = torch.nn.Parameter(torch.randn(5, 3))
    old_rows = [0, 1, 2]
    _register_freeze_hook(w, old_rows)
    (w.sum()).backward()
    g = w.grad
    assert g is not None
    assert torch.allclose(g[old_rows], torch.zeros_like(g[old_rows])), \
        f"old rows not zeroed: {g[old_rows]}"
    assert not torch.allclose(g[3:], torch.zeros_like(g[3:])), \
        f"new rows wrongly zeroed: {g[3:]}"
    print("[row-freeze self-check] OK: 5-row param, 3 old rows zeroed, 2 new rows trained.")


def _read_manifest(path: str) -> list[str]:
    """One path per line, skip blanks/comments. Same format the orchestrator writes."""
    out: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _main_cli() -> None:
    """Standalone CLI entrypoint for running on a RunPod pod.

    The orchestrator uploads this file + the codec JSON + base model + train/val
    manifests, then runs `python3 ketos_trainer_api.py -- ...` on the pod.
    """
    import argparse
    p = argparse.ArgumentParser(description="Kraken row-freeze training harness (Phase 0b/§1)")
    p.add_argument("--load", required=True, help="Base .safetensors or .mlmodel path")
    p.add_argument("--train-manifest", required=True, help="Text file: one XML path per line")
    p.add_argument("--eval-manifest", default=None, help="Optional eval manifest (same format)")
    p.add_argument("--codec-json", required=True, help="Phase 0a union codec JSON (row map)")
    p.add_argument("--checkpoint-path", default="model", help="Output dir for checkpoints")
    p.add_argument("--weights-format", default="safetensors", choices=["safetensors", "coreml"])
    p.add_argument("--format-type", default="page", choices=["path", "xml", "alto", "page", "binary"])
    p.add_argument("--resize", default="union", choices=["fail", "union", "new"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min-epochs", type=int, default=8)
    p.add_argument("--lag", type=int, default=25)
    p.add_argument("--lrate", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--quit", default="early", choices=["fixed", "early", "dilate"])
    p.add_argument("--freeze-backbone", type=int, default=999999)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--schedule", default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--precision", default="32-true")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--reorder", default="rtl")
    p.add_argument("--freq", type=float, default=1.0)
    p.add_argument("--no-freeze-old-rows", dest="freeze_old_rows", action="store_false",
                   help="Disable the §1 row freeze (train all classifier rows)")
    p.set_defaults(freeze_old_rows=True)
    args = p.parse_args()

    cfg = RowFreezeConfig(
        load_model=args.load,
        train_xml_files=_read_manifest(args.train_manifest),
        eval_xml_files=_read_manifest(args.eval_manifest) if args.eval_manifest else None,
        format_type=args.format_type,
        resize=args.resize,
        checkpoint_path=args.checkpoint_path,
        weights_format=args.weights_format,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        lag=args.lag,
        lrate=args.lrate,
        warmup=args.warmup,
        quit=args.quit,
        freeze_backbone=args.freeze_backbone,
        augment=args.augment,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        schedule=args.schedule,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        reorder=args.reorder,
        freq=args.freq,
        freeze_old_rows=args.freeze_old_rows,
        codec_json=args.codec_json,
    )
    out = train_with_row_freeze(cfg)
    print(f"[row-freeze] best checkpoint: {out}")


if __name__ == "__main__":
    # `python -m msocr.training.ketos_trainer_api` runs the self-check (no args).
    # `python ketos_trainer_api.py -- ...` runs the CLI (with args).
    import sys
    if len(sys.argv) > 1:
        _main_cli()
    else:
        _self_check()