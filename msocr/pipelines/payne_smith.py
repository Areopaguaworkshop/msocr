"""Payne-Smith Syriac OCR pipeline (phases 0-8).

YAML-first runner that maps phases to concrete actions.
Default behavior is dry-run unless execute=True.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from msocr.preprocessing.sauvola import preprocess_directory as sauvola_preprocess
from msocr.segmentation.kraken_blla import segment_pages
from msocr.segmentation.line_extraction import extract_lines_from_segments
from msocr.segmentation.region_classification import write_region_labels
from msocr.training.ketos_runner import ketos_train_xml
from msocr.training.bootstrap_ocr import bootstrap_ocr_lines
from msocr.datasets.splitter import split_dataset


@dataclass
class PhaseResult:
    phase: str
    ok: bool
    details: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cmd: List[str], execute: bool, cwd: Optional[Path] = None) -> PhaseResult:
    joined = " ".join(cmd)
    if not execute:
        return PhaseResult("command", True, f"DRY-RUN: {joined}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return PhaseResult("command", True, joined)
    except subprocess.CalledProcessError as exc:
        return PhaseResult("command", False, f"FAILED: {joined} ({exc})")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class PayneSmithPipeline:
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
        self.config = _load_yaml(config_path)
        self.runpod = _load_yaml(runpod_path) if runpod_path else None

    def run(self, phases: Iterable[str], input_pdf: Optional[Path] = None) -> List[PhaseResult]:
        results: List[PhaseResult] = []
        phase_map = {
            "0": self.phase_0_pretrain,
            "0b": self.phase_0b_synthetic_augmentation,
            "1": lambda: self.phase_1_ingest_pages(input_pdf),
            "2": self.phase_2_preprocess,
            "3": self.phase_3_segmentation,
            "3b": self.phase_3b_region_classification,
            "4": self.phase_4_line_extraction,
            "5": self.phase_5_bootstrap_ocr,
            "6": self.phase_6_manual_correction,
            "7": self.phase_7_dataset_split,
            "8": self.phase_8_training,
        }
        for phase in phases:
            if phase not in phase_map:
                results.append(PhaseResult(phase, False, "Unknown phase"))
                continue
            results.append(phase_map[phase]())
        return results

    def phase_0_pretrain(self) -> PhaseResult:
        base = self.workdir
        dataset_dir = _resolve("dataset/pretrain/vienna", base)
        model_out = _resolve("models/pretrain/vienna_serto", base)
        xml_glob = str(dataset_dir / "**/*.xml")
        cmd = [
            "ketos",
            "train",
            "-f",
            "xml",
            "--base-dir",
            "R",
            "--augment",
            "--lag",
            "10",
            "--min-epochs",
            "30",
            "--device",
            "cuda:0",
            "--output",
            str(model_out),
            xml_glob,
        ]
        result = _run(cmd, self.execute)
        result.phase = "phase_0_pretrain"
        return result

    def phase_0b_synthetic_augmentation(self) -> PhaseResult:
        cfg = self.config["pipeline"]["phase_0b_synthetic_augmentation"]
        base = self.workdir
        out_dir = _resolve(cfg["output"], base)
        _ensure_dir(out_dir)
        cmd = [
            "trdg",
            "--language",
            "syr",
            "--fonts-dir",
            "./fonts/syriac",
            "--count",
            str(cfg.get("line_count", 2000)),
            "--output-dir",
            str(out_dir),
            "--margins",
            "5,5,5,5",
            "--blur",
            "1",
            "--distorsion",
            "1",
            "--random-blur",
            "--format",
            str(self.config["configuration"].get("line_height", 48)),
        ]
        result = _run(cmd, self.execute)
        result.phase = "phase_0b_synthetic_augmentation"
        return result

    def phase_1_ingest_pages(self, input_pdf: Optional[Path]) -> PhaseResult:
        if not input_pdf:
            return PhaseResult("phase_1_ingest_pages", False, "input_pdf is required")
        base = self.workdir
        raw_dir = _resolve(self.config["directories"]["dataset"]["raw_pages"], base)
        _ensure_dir(raw_dir)
        dpi = str(self.config["configuration"].get("dpi", 500))
        cmd = [
            "pdftoppm",
            "-tiff",
            "-r",
            dpi,
            str(input_pdf),
            str(raw_dir / "page"),
        ]
        result = _run(cmd, self.execute)
        result.phase = "phase_1_ingest_pages"
        return result

    def phase_2_preprocess(self) -> PhaseResult:
        base = self.workdir
        raw_dir = _resolve(self.config["directories"]["dataset"]["raw_pages"], base)
        processed_dir = _resolve(self.config["directories"]["dataset"]["processed_pages"], base)
        _ensure_dir(processed_dir)
        cfg = self.config["configuration"]["preprocessing"]
        if not self.execute:
            return PhaseResult("phase_2_preprocess", True, f"DRY-RUN: preprocess {raw_dir} -> {processed_dir}")
        count = sauvola_preprocess(raw_dir, processed_dir, cfg)
        return PhaseResult("phase_2_preprocess", True, f"processed={count}")

    def phase_3_segmentation(self) -> PhaseResult:
        base = self.workdir
        processed_dir = _resolve(self.config["directories"]["dataset"]["processed_pages"], base)
        seg_dir = _resolve("dataset/segments", base)
        _ensure_dir(seg_dir)
        if not self.execute:
            return PhaseResult("phase_3_segmentation", True, f"DRY-RUN: segment {processed_dir} -> {seg_dir}")
        cfg = self.config["configuration"]["segmentation"]
        count = segment_pages(processed_dir, seg_dir, cfg)
        return PhaseResult("phase_3_segmentation", True, f"segments={count}")

    def phase_3b_region_classification(self) -> PhaseResult:
        base = self.workdir
        seg_dir = _resolve("dataset/segments", base)
        labels_path = _resolve("dataset/lines/region_labels.json", base)
        _ensure_dir(labels_path.parent)
        if not self.execute:
            return PhaseResult("phase_3b_region_classification", True, f"DRY-RUN: region labels -> {labels_path}")
        count = write_region_labels(seg_dir, labels_path, self.config["pipeline"]["phase_3b_region_classification"]["region_types"])
        return PhaseResult("phase_3b_region_classification", True, f"labels={count}")

    def phase_4_line_extraction(self) -> PhaseResult:
        base = self.workdir
        processed_dir = _resolve(self.config["directories"]["dataset"]["processed_pages"], base)
        seg_dir = _resolve("dataset/segments", base)
        lines_dir = _resolve(self.config["directories"]["dataset"]["lines"], base)
        _ensure_dir(lines_dir)
        if not self.execute:
            return PhaseResult("phase_4_line_extraction", True, f"DRY-RUN: extract lines -> {lines_dir}")
        count = extract_lines_from_segments(processed_dir, seg_dir, lines_dir)
        return PhaseResult("phase_4_line_extraction", True, f"lines={count}")

    def phase_5_bootstrap_ocr(self) -> PhaseResult:
        base = self.workdir
        lines_dir = _resolve(self.config["directories"]["dataset"]["lines"], base)
        bootstrap_dir = _resolve(self.config["directories"]["dataset"]["bootstrap"], base)
        rejected_dir = _resolve(self.config["directories"]["dataset"]["rejected"], base)
        _ensure_dir(bootstrap_dir)
        _ensure_dir(rejected_dir)
        model_path = _resolve(self.config["configuration"]["bootstrap"]["model"], base)
        cfg = self.config["configuration"]["bootstrap"]
        if not self.execute:
            return PhaseResult("phase_5_bootstrap_ocr", True, f"DRY-RUN: bootstrap OCR {lines_dir} -> {bootstrap_dir}")
        count = bootstrap_ocr_lines(lines_dir, model_path, bootstrap_dir, rejected_dir, cfg)
        return PhaseResult("phase_5_bootstrap_ocr", True, f"bootstrapped={count}")

    def phase_6_manual_correction(self) -> PhaseResult:
        return PhaseResult(
            "phase_6_manual_correction",
            True,
            "Manual step: import bootstrap output into eScriptorium, correct, export PAGE XML to dataset/corrected_lines",
        )

    def phase_7_dataset_split(self) -> PhaseResult:
        base = self.workdir
        corrected_dir = _resolve(self.config["directories"]["dataset"]["corrected"], base)
        train_dir = _resolve(self.config["directories"]["dataset"]["train"], base)
        val_dir = _resolve(self.config["directories"]["dataset"]["validation"], base)
        hold_dir = _resolve(self.config["directories"]["dataset"]["holdout"], base)
        if not self.execute:
            return PhaseResult("phase_7_dataset_split", True, "DRY-RUN: split corrected dataset")
        cfg = self.config["configuration"]["dataset"]
        split_dataset(corrected_dir, train_dir, val_dir, hold_dir, cfg)
        return PhaseResult("phase_7_dataset_split", True, "split complete")

    def phase_8_training(self) -> PhaseResult:
        base = self.workdir
        train_dir = _resolve(self.config["directories"]["dataset"]["train"], base)
        val_dir = _resolve(self.config["directories"]["dataset"]["validation"], base)
        model_out = _resolve("models/finetune/paynesmith_serto_v1", base)
        if not self.execute:
            return PhaseResult("phase_8_training", True, "DRY-RUN: ketos train for Serto/Estrangela")
        serto_train = train_dir / "serto"
        serto_val = val_dir / "serto"
        ketos_train_xml(
            train_glob=str(serto_train / "*.xml"),
            eval_glob=str(serto_val / "*.xml"),
            output_prefix=str(model_out),
            device="cuda:0",
        )
        return PhaseResult("phase_8_training", True, "training invoked")
