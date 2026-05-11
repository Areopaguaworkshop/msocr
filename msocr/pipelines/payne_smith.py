"""Payne-Smith Syriac OCR pipeline runner.

Supports both legacy phase shortcuts (0..8) and the full staged phase names
from `pipeline/payne-smith_syriac_runpod_train.yaml`.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import yaml
from PIL import Image

from msocr.datasets.paynesmith_split import split_paynesmith_dataset
from msocr.preprocessing.sauvola import preprocess_directory as sauvola_preprocess
from msocr.segmentation.line_extraction import extract_lines_from_segments
from msocr.segmentation.kraken_blla import segment_pages
from msocr.segmentation.region_classification import write_region_labels
from msocr.training.bootstrap_ocr import bootstrap_ocr_lines
from msocr.training.ketos_runner import ketos_train_xml
from msocr.training.synthetic_generator import generate_synthetic_lines
from msocr.utils.language_correction import correct_ocr_directory


@dataclass
class PhaseResult:
    phase: str
    ok: bool
    details: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run(cmd: list[str], execute: bool, phase: str, cwd: Optional[Path] = None) -> PhaseResult:
    joined = " ".join(cmd)
    if not execute:
        return PhaseResult(phase, True, f"DRY-RUN: {joined}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return PhaseResult(phase, True, joined)
    except subprocess.CalledProcessError as exc:
        return PhaseResult(phase, False, f"FAILED: {joined} ({exc})")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class PayneSmithPipeline:
    DEFAULT_PHASES = [
        "phase_1a_validate_vienna_gt",
        "phase_1b_split_vienna_dataset",
        "phase_1c_pretrain",
        "phase_1d_evaluate_vienna",
        "phase_2a_ingest_pages",
        "phase_2b_synthetic_augmentation",
        "phase_2c_preprocess",
        "phase_2d_segmentation",
        "phase_2e_line_extraction",
        "phase_2f_bootstrap_ocr",
        "phase_2g_manual_correction",
        "phase_2h_dataset_split",
        "phase_2i_finetune_serto",
        "phase_2j_finetune_estrangela",
        "phase_2l_language_correction",
        "stage_3_evaluation",
    ]

    PHASE_ALIASES = {
        "0": "phase_1c_pretrain",
        "0b": "phase_2b_synthetic_augmentation",
        "1": "phase_2a_ingest_pages",
        "2": "phase_2c_preprocess",
        "3": "phase_2d_segmentation",
        "3b": "phase_2d_segmentation",
        "4": "phase_2e_line_extraction",
        "5": "phase_2f_bootstrap_ocr",
        "6": "phase_2g_manual_correction",
        "7": "phase_2h_dataset_split",
        "8": "phase_2i_finetune_serto",
        "2i": "phase_2i_finetune_serto",
        "2j": "phase_2j_finetune_estrangela",
        "2k": "phase_2k_incremental_training",
        "2l": "phase_2l_language_correction",
        "3eval": "stage_3_evaluation",
        "stage_3": "stage_3_evaluation",
    }

    def __init__(
        self,
        config_path: Path,
        runpod_path: Optional[Path] = None,
        workdir: Optional[Path] = None,
        execute: bool = False,
    ) -> None:
        self.config_path = config_path
        self.runpod_path = runpod_path
        self.workdir = workdir or Path.cwd()
        self.execute = execute
        self.config = _load_yaml(config_path) if config_path and config_path.exists() else {}
        self.runpod = _load_yaml(runpod_path) if runpod_path and runpod_path.exists() else {}
        if not self.runpod and self.config.get("stage_1_vienna_pretrain"):
            self.runpod = self.config
        self.settings = self.runpod or self.config
        if not self.settings:
            raise ValueError("No valid Payne-Smith pipeline config found.")
        self.paths = self._build_paths()
        self.input_pdf: Optional[Path] = None

    def _build_paths(self) -> dict[str, Path]:
        base = self.workdir
        ds = self.config.get("directories", {}).get("dataset", {})

        train_root = _resolve(ds.get("train", "dataset/train"), base)
        validation_root = _resolve(ds.get("validation", "dataset/validation"), base)

        return {
            "vienna": _resolve(ds.get("vienna", "dataset/vienna"), base),
            "pretrain_train": _resolve("dataset/pretrain/train", base),
            "pretrain_validation": _resolve("dataset/pretrain/validation", base),
            "pretrain_holdout": _resolve("dataset/pretrain/holdout", base),
            "raw_pages": _resolve(ds.get("raw_pages", "dataset/raw_pages"), base),
            "processed_pages": _resolve(ds.get("processed_pages", "dataset/processed_pages"), base),
            "segments": _resolve("dataset/segments", base),
            "lines": _resolve(ds.get("lines", "dataset/lines"), base),
            "region_labels": _resolve("dataset/lines/region_labels.json", base),
            "bootstrap": _resolve(ds.get("bootstrap", "dataset/bootstrap"), base),
            "rejected": _resolve(ds.get("rejected", "dataset/rejected"), base),
            "corrected": _resolve(ds.get("corrected", "dataset/corrected"), base),
            "synthetic": _resolve(ds.get("synthetic", "dataset/synthetic_lines"), base),
            "train_serto": train_root / "serto",
            "train_estrangela": train_root / "estrangela",
            "validation_serto": validation_root / "serto",
            "validation_estrangela": validation_root / "estrangela",
            "holdout": _resolve(ds.get("holdout", "dataset/holdout"), base),
            "split_manifests": _resolve("dataset/split_manifests", base),
            "pretrain_dir": _resolve("models/pretrain", base),
            "finetune_dir": _resolve("models/finetune", base),
            "exports_raw_ocr": _resolve("exports/raw_ocr", base),
            "exports_corrected_ocr": _resolve("exports/corrected_ocr", base),
            "lexicon_dir": _resolve("scripts/lexicons", base),
            "input_dir": _resolve("input", base),
        }

    def _configuration(self) -> Dict[str, Any]:
        return self.settings.get("configuration", {})

    def _cfg(self, key: str, default: Any) -> Any:
        return self._configuration().get(key, default)

    def _normalize_phase(self, phase: str) -> str:
        p = phase.strip()
        if p in self.PHASE_ALIASES:
            return self.PHASE_ALIASES[p]
        if p == "all":
            return "all"
        return p

    def _find_pretrain_model(self) -> Path:
        configured = self._configuration().get("bootstrap", {}).get("model", "models/pretrain/vienna_serto.mlmodel")
        candidate = _resolve(str(configured), self.workdir)
        if candidate.exists():
            return candidate
        alt = self.paths["pretrain_dir"] / "vienna_serto.mlmodel"
        if alt.exists():
            return alt
        return candidate

    def _find_latest_serto_model(self) -> Path:
        candidates = sorted(self.paths["finetune_dir"].glob("paynesmith_serto*.mlmodel"))
        if candidates:
            return candidates[-1]
        return self.paths["finetune_dir"] / "paynesmith_serto_v1.mlmodel"

    def run(self, phases: Iterable[str], input_pdf: Optional[Path] = None) -> list[PhaseResult]:
        results: list[PhaseResult] = []
        self.input_pdf = input_pdf
        phase_map: dict[str, Callable[[], PhaseResult]] = {
            "phase_1a_validate_vienna_gt": self.phase_1a_validate_vienna_gt,
            "phase_1b_split_vienna_dataset": self.phase_1b_split_vienna_dataset,
            "phase_1c_pretrain": self.phase_1c_pretrain,
            "phase_1d_evaluate_vienna": self.phase_1d_evaluate_vienna,
            "phase_2a_ingest_pages": self.phase_2a_ingest_pages,
            "phase_2b_synthetic_augmentation": self.phase_2b_synthetic_augmentation,
            "phase_2c_preprocess": self.phase_2c_preprocess,
            "phase_2d_segmentation": self.phase_2d_segmentation,
            "phase_2e_line_extraction": self.phase_2e_line_extraction,
            "phase_2f_bootstrap_ocr": self.phase_2f_bootstrap_ocr,
            "phase_2g_manual_correction": self.phase_2g_manual_correction,
            "phase_2h_dataset_split": self.phase_2h_dataset_split,
            "phase_2i_finetune_serto": self.phase_2i_finetune_serto,
            "phase_2j_finetune_estrangela": self.phase_2j_finetune_estrangela,
            "phase_2k_incremental_training": self.phase_2k_incremental_training,
            "phase_2l_language_correction": self.phase_2l_language_correction,
            "stage_3_evaluation": self.stage_3_evaluation,
        }

        for phase in phases:
            normalized = self._normalize_phase(phase)
            if normalized == "all":
                for full_phase in self.DEFAULT_PHASES:
                    results.append(phase_map[full_phase]())
                continue
            if normalized not in phase_map:
                results.append(PhaseResult(phase, False, "Unknown phase"))
                continue
            results.append(phase_map[normalized]())
        return results

    def phase_1a_validate_vienna_gt(self) -> PhaseResult:
        dataset_dir = self.paths["vienna"]
        if not self.execute:
            return PhaseResult(
                "phase_1a_validate_vienna_gt",
                True,
                f"DRY-RUN: validate {dataset_dir}",
            )
        xml_files = sorted(dataset_dir.rglob("*.xml"))
        if not xml_files:
            return PhaseResult("phase_1a_validate_vienna_gt", False, f"No XML files in {dataset_dir}")
        missing_images = []
        for xml_file in xml_files:
            if not any(xml_file.with_suffix(ext).exists() for ext in (".png", ".jpg", ".tif", ".tiff")):
                missing_images.append(xml_file.name)
        details = f"xml={len(xml_files)} missing_images={len(missing_images)}"
        return PhaseResult("phase_1a_validate_vienna_gt", True, details)

    def phase_1b_split_vienna_dataset(self) -> PhaseResult:
        if not self.execute:
            return PhaseResult("phase_1b_split_vienna_dataset", True, "DRY-RUN: split Vienna dataset")

        xml_files = sorted(self.paths["vienna"].rglob("*.xml"))
        if not xml_files:
            return PhaseResult("phase_1b_split_vienna_dataset", False, "No Vienna XML files found")

        seed = int(self._cfg("random_seed", 42))
        train_ratio = float(self._cfg("dataset", {}).get("vienna_train_ratio", 0.85))
        validation_ratio = float(self._cfg("dataset", {}).get("vienna_validation_ratio", 0.10))
        holdout_ratio = float(self._cfg("dataset", {}).get("vienna_holdout_ratio", 0.05))

        rng = random.Random(seed)
        rng.shuffle(xml_files)
        n = len(xml_files)
        n_holdout = max(1, int(n * holdout_ratio))
        n_validation = max(1, int(n * validation_ratio))
        n_train = max(0, n - n_holdout - n_validation)

        split_map = {
            "train": xml_files[:n_train],
            "validation": xml_files[n_train : n_train + n_validation],
            "holdout": xml_files[n_train + n_validation :],
        }

        targets = {
            "train": self.paths["pretrain_train"],
            "validation": self.paths["pretrain_validation"],
            "holdout": self.paths["pretrain_holdout"],
        }
        for target in targets.values():
            _ensure_dir(target)
        for name, files in split_map.items():
            for xml_file in files:
                shutil.copy2(xml_file, targets[name] / xml_file.name)
                for ext in (".png", ".jpg", ".tif", ".tiff"):
                    image = xml_file.with_suffix(ext)
                    if image.exists():
                        shutil.copy2(image, targets[name] / image.name)
                        break

        manifest = {
            "seed": seed,
            "ratios": {"train": train_ratio, "validation": validation_ratio, "holdout": holdout_ratio},
            "total": n,
            "train": [f.name for f in split_map["train"]],
            "validation": [f.name for f in split_map["validation"]],
            "holdout": [f.name for f in split_map["holdout"]],
        }
        _ensure_dir(self.paths["split_manifests"])
        out_manifest = self.paths["split_manifests"] / "vienna_split.json"
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return PhaseResult(
            "phase_1b_split_vienna_dataset",
            True,
            f"train={len(split_map['train'])} val={len(split_map['validation'])} holdout={len(split_map['holdout'])}",
        )

    def phase_1c_pretrain(self) -> PhaseResult:
        model_out = self.paths["pretrain_dir"] / "vienna_serto"
        train_glob = str(self.paths["pretrain_train"] / "*.xml")
        val_glob = str(self.paths["pretrain_validation"] / "*.xml")
        if not self.execute:
            return PhaseResult("phase_1c_pretrain", True, f"DRY-RUN: ketos train {train_glob}")
        _ensure_dir(self.paths["pretrain_dir"])
        ketos_train_xml(
            train_globs=[train_glob],
            eval_glob=val_glob,
            output_prefix=str(model_out),
            device="cuda:0",
            min_epochs=20,
            lag=10,
            augment=True,
        )
        return PhaseResult("phase_1c_pretrain", True, f"output={model_out}.mlmodel")

    def phase_1d_evaluate_vienna(self) -> PhaseResult:
        model_path = self._find_pretrain_model()
        cmd = [
            "kraken",
            "-d",
            "cuda:0",
            "test",
            "--model",
            str(model_path),
            "--evaluation-files",
            str(self.paths["pretrain_holdout"] / "*.xml"),
            "--text-direction",
            "rtl",
        ]
        return _run(cmd, self.execute, phase="phase_1d_evaluate_vienna")

    def phase_2a_ingest_pages(self) -> PhaseResult:
        input_pdf = self.input_pdf or (self.paths["input_dir"] / "payne_smith.pdf")
        if not input_pdf.exists() and not self.execute:
            return PhaseResult(
                "phase_2a_ingest_pages",
                True,
                f"DRY-RUN: input PDF expected at {input_pdf} (or pass --input-pdf)",
            )
        if not input_pdf.exists():
            return PhaseResult("phase_2a_ingest_pages", False, f"Input PDF not found: {input_pdf}")

        _ensure_dir(self.paths["raw_pages"])
        dpi = int(self._cfg("dpi", 500))
        cmd = ["pdftoppm", "-tiff", "-r", str(dpi), str(input_pdf), str(self.paths["raw_pages"] / "page")]
        pre = _run(cmd, self.execute, phase="phase_2a_ingest_pages")
        if not self.execute or not pre.ok:
            return pre

        converted = 0
        for tif_path in sorted(self.paths["raw_pages"].glob("*.tif")):
            png_path = tif_path.with_suffix(".png")
            with Image.open(tif_path) as image:
                image.save(png_path)
            tif_path.unlink(missing_ok=True)
            converted += 1
        return PhaseResult("phase_2a_ingest_pages", True, f"converted_pages={converted}")

    def phase_2b_synthetic_augmentation(self) -> PhaseResult:
        output_dir = self.paths["synthetic"]
        count = 2000
        seed = int(self._cfg("random_seed", 42))
        line_height = int(self._cfg("line_height", 48))
        if not self.execute:
            return PhaseResult(
                "phase_2b_synthetic_augmentation",
                True,
                f"DRY-RUN: generate synthetic lines -> {output_dir} count={count}",
            )
        generated = generate_synthetic_lines(
            output_dir=output_dir,
            count=count,
            line_height=line_height,
            seed=seed,
            fonts=["estrangelo", "serto", "syriac"],
            augment=True,
        )
        return PhaseResult("phase_2b_synthetic_augmentation", True, f"generated={generated}")

    def phase_2c_preprocess(self) -> PhaseResult:
        cfg = self._cfg("preprocessing", {})
        if not self.execute:
            return PhaseResult(
                "phase_2c_preprocess",
                True,
                f"DRY-RUN: preprocess {self.paths['raw_pages']} -> {self.paths['processed_pages']}",
            )
        _ensure_dir(self.paths["processed_pages"])
        count = sauvola_preprocess(self.paths["raw_pages"], self.paths["processed_pages"], cfg)
        return PhaseResult("phase_2c_preprocess", True, f"processed={count}")

    def phase_2d_segmentation(self) -> PhaseResult:
        cfg = self._cfg("segmentation", {})
        if not self.execute:
            return PhaseResult(
                "phase_2d_segmentation",
                True,
                f"DRY-RUN: segment {self.paths['processed_pages']} -> {self.paths['segments']}",
            )
        _ensure_dir(self.paths["segments"])
        count = segment_pages(self.paths["processed_pages"], self.paths["segments"], cfg)
        region_map = {
            "MainZone": "serto",
            "TitleZone": "estrangela",
            "MarginZone": "serto",
        }
        labels = write_region_labels(self.paths["segments"], self.paths["region_labels"], region_map)
        return PhaseResult("phase_2d_segmentation", True, f"segments={count} region_labels={labels}")

    def phase_2e_line_extraction(self) -> PhaseResult:
        if not self.execute:
            return PhaseResult(
                "phase_2e_line_extraction",
                True,
                f"DRY-RUN: extract lines -> {self.paths['lines']}",
            )
        _ensure_dir(self.paths["lines"])
        count = extract_lines_from_segments(self.paths["processed_pages"], self.paths["segments"], self.paths["lines"])
        return PhaseResult("phase_2e_line_extraction", True, f"lines={count}")

    def phase_2f_bootstrap_ocr(self) -> PhaseResult:
        model_path = self._find_pretrain_model()
        cfg = self._cfg("bootstrap", {"confidence_aggregation": "mean", "confidence_threshold": 0.80})
        _ensure_dir(self.paths["bootstrap"])
        _ensure_dir(self.paths["rejected"])
        if not self.execute:
            return PhaseResult(
                "phase_2f_bootstrap_ocr",
                True,
                f"DRY-RUN: bootstrap with {model_path} -> {self.paths['bootstrap']} / {self.paths['rejected']}",
            )
        count = bootstrap_ocr_lines(
            self.paths["lines"],
            model_path,
            self.paths["bootstrap"],
            self.paths["rejected"],
            cfg,
        )
        return PhaseResult("phase_2f_bootstrap_ocr", True, f"processed_lines={count}")

    def phase_2g_manual_correction(self) -> PhaseResult:
        return PhaseResult(
            "phase_2g_manual_correction",
            True,
            "Manual step: correct bootstrap outputs in eScriptorium and export PAGE XML to dataset/corrected",
        )

    def phase_2h_dataset_split(self) -> PhaseResult:
        dataset_cfg = self._cfg("dataset", {})
        seed = int(self._cfg("random_seed", 42))
        if not self.execute:
            return PhaseResult("phase_2h_dataset_split", True, "DRY-RUN: stratified Payne-Smith split")

        summary = split_paynesmith_dataset(
            corrected_dir=self.paths["corrected"],
            train_serto_dir=self.paths["train_serto"],
            train_estrangela_dir=self.paths["train_estrangela"],
            validation_serto_dir=self.paths["validation_serto"],
            validation_estrangela_dir=self.paths["validation_estrangela"],
            holdout_dir=self.paths["holdout"],
            manifest_path=self.paths["split_manifests"] / "paynesmith_split.json",
            seed=seed,
            train_ratio=float(dataset_cfg.get("train", 0.85)),
            validation_ratio=float(dataset_cfg.get("paynesmith_validation_ratio", 0.10)),
            holdout_ratio=float(dataset_cfg.get("paynesmith_holdout_ratio", 0.05)),
            random_sample_ratio=0.60,
            uncertainty_sample_ratio=float(dataset_cfg.get("active_learning_ratio", 0.40)),
            region_labels_path=self.paths["region_labels"],
            bootstrap_dir=self.paths["bootstrap"],
            rejected_dir=self.paths["rejected"],
        )
        return PhaseResult(
            "phase_2h_dataset_split",
            True,
            (
                f"train={summary.get('train', 0)} val={summary.get('validation', 0)} "
                f"holdout={summary.get('holdout', 0)}"
            ),
        )

    def phase_2i_finetune_serto(self) -> PhaseResult:
        output_prefix = self.paths["finetune_dir"] / "paynesmith_serto_v1"
        train_globs = [str(self.paths["train_serto"] / "*.xml"), str(self.paths["synthetic"] / "*.xml")]
        eval_glob = str(self.paths["validation_serto"] / "*.xml")
        if not self.execute:
            return PhaseResult("phase_2i_finetune_serto", True, f"DRY-RUN: train Serto model -> {output_prefix}")
        _ensure_dir(self.paths["finetune_dir"])
        ketos_train_xml(
            train_globs=train_globs,
            eval_glob=eval_glob,
            output_prefix=str(output_prefix),
            device="cuda:0",
            min_epochs=15,
            lag=10,
            augment=True,
        )
        return PhaseResult("phase_2i_finetune_serto", True, f"output={output_prefix}.mlmodel")

    def phase_2j_finetune_estrangela(self) -> PhaseResult:
        estrangela_xml = sorted(self.paths["train_estrangela"].glob("*.xml"))
        out_model = self.paths["finetune_dir"] / "paynesmith_estrangela_v1.mlmodel"
        if len(estrangela_xml) < 50:
            if not self.execute:
                return PhaseResult(
                    "phase_2j_finetune_estrangela",
                    True,
                    f"DRY-RUN: fallback copy from Vienna base model (<50 lines, found {len(estrangela_xml)})",
                )
            source_model = self._find_pretrain_model()
            _ensure_dir(self.paths["finetune_dir"])
            if source_model.exists():
                shutil.copy2(source_model, out_model)
                return PhaseResult(
                    "phase_2j_finetune_estrangela",
                    True,
                    f"fallback_used source={source_model} target={out_model}",
                )
            return PhaseResult("phase_2j_finetune_estrangela", False, "Fallback source model not found")

        output_prefix = self.paths["finetune_dir"] / "paynesmith_estrangela_v1"
        if not self.execute:
            return PhaseResult("phase_2j_finetune_estrangela", True, f"DRY-RUN: train Estrangela model -> {output_prefix}")
        ketos_train_xml(
            train_globs=[str(self.paths["train_estrangela"] / "*.xml")],
            eval_glob=str(self.paths["validation_estrangela"] / "*.xml"),
            output_prefix=str(output_prefix),
            device="cuda:0",
            min_epochs=10,
            lag=10,
            augment=True,
        )
        return PhaseResult("phase_2j_finetune_estrangela", True, f"output={output_prefix}.mlmodel")

    def phase_2k_incremental_training(self) -> PhaseResult:
        prev_model = self._find_latest_serto_model()
        train_files = [str(p) for p in sorted(self.paths["train_serto"].glob("*.xml"))]
        val_files = [str(p) for p in sorted(self.paths["validation_serto"].glob("*.xml"))]
        if not self.execute:
            return PhaseResult(
                "phase_2k_incremental_training",
                True,
                f"DRY-RUN: incremental train from {prev_model}",
            )
        if not prev_model.exists():
            return PhaseResult("phase_2k_incremental_training", False, f"Previous model not found: {prev_model}")
        if not train_files:
            return PhaseResult("phase_2k_incremental_training", False, "No Serto train files found")

        output_prefix = self.paths["finetune_dir"] / "paynesmith_serto_incremental"
        cmd = [
            "kraken",
            "-d",
            "cuda:0",
            "train",
            "--model",
            str(prev_model),
            "--ground-truth",
            *train_files,
            "--evaluation-files",
            *val_files,
            "--output",
            str(output_prefix),
            "--resize",
            "add",
            "--text-direction",
            "rtl",
            "--augment",
            "--lag",
            "10",
        ]
        return _run(cmd, self.execute, phase="phase_2k_incremental_training")

    def phase_2l_language_correction(self) -> PhaseResult:
        if not self.execute:
            return PhaseResult(
                "phase_2l_language_correction",
                True,
                f"DRY-RUN: language correction {self.paths['exports_raw_ocr']} -> {self.paths['exports_corrected_ocr']}",
            )
        corrected = correct_ocr_directory(
            input_dir=self.paths["exports_raw_ocr"],
            output_dir=self.paths["exports_corrected_ocr"],
            syriac_lexicon_path=self.paths["lexicon_dir"] / "sedra.txt",
            latin_lexicon_path=self.paths["lexicon_dir"] / "latin_en.txt",
            max_edit_distance=int(self._cfg("language_correction", {}).get("symspell_max_edit_distance", 1)),
            syriac_range=(0x0700, 0x074F),
        )
        return PhaseResult("phase_2l_language_correction", True, f"corrected_files={corrected}")

    def stage_3_evaluation(self) -> PhaseResult:
        holdout_files = sorted(self.paths["holdout"].glob("*.xml"))
        model_path = self._find_latest_serto_model()
        if not self.execute:
            return PhaseResult(
                "stage_3_evaluation",
                True,
                f"DRY-RUN: evaluate {model_path} on {self.paths['holdout']}",
            )
        if not holdout_files:
            return PhaseResult("stage_3_evaluation", False, "No holdout XML files found")

        cmd = [
            "kraken",
            "-d",
            "cuda:0",
            "test",
            "--model",
            str(model_path),
            "--evaluation-files",
            *[str(p) for p in holdout_files],
            "--text-direction",
            "rtl",
        ]
        return _run(cmd, self.execute, phase="stage_3_evaluation")
