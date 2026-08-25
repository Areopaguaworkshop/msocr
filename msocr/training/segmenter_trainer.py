"""Prepare and run the four fragment-layout candidate trainers on RunPod."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from lxml import etree
from PIL import Image

from msocr.data.manifest import (
    ManifestCase,
    iter_style_group_cases,
    load_frozen_manifest,
)
from msocr.training.runpod_runner import RunPodRunner

BACKENDS = ("blla", "orli", "dfine", "yolo-obb")
_REMOTE_ROOT = "/workspace/job"
_SETUP = {
    "blla": "python3 -m pip install --quiet kraken==7.0.2",
    "orli": "python3 -m pip install --quiet orli==0.0.2",
    "dfine": "python3 -m pip install --quiet dfine_kraken==0.4.3",
    "yolo-obb": "python3 -m pip install --quiet ultralytics==8.4.82",
}


@dataclass(frozen=True)
class SegmenterJob:
    backend: str
    name: str
    archive: Path
    train_cmd: list[str]
    setup_cmds: list[str]
    artifact_remote_dir: str
    artifact_pattern: str
    artifact_remote_path: str | None
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _points(value: str | None, source: Path) -> list[tuple[float, float]]:
    if not value:
        return []
    try:
        points: list[tuple[float, float]] = []
        for point in value.split():
            raw_coordinates = point.split(",")
            if len(raw_coordinates) != 2:
                raise ValueError
            coordinates = (float(raw_coordinates[0]), float(raw_coordinates[1]))
            if not all(map(math.isfinite, coordinates)):
                raise ValueError
            points.append(coordinates)
        return points
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PAGE points in {source}: {value!r}") from exc


def _find_child(element: etree._Element, local_name: str) -> etree._Element | None:
    return next(
        (child for child in element if etree.QName(child).localname == local_name),
        None,
    )


def _line_box(line: etree._Element, source: Path, line_height: float) -> np.ndarray:
    coords = _find_child(line, "Coords")
    coords_points = _points(
        coords.get("points") if coords is not None else None, source
    )
    if len(coords_points) >= 3:
        rect = cv2.minAreaRect(np.asarray(coords_points, dtype=np.float32))
        return cv2.boxPoints(rect)

    baseline = _find_child(line, "Baseline")
    baseline_points = _points(
        baseline.get("points") if baseline is not None else None, source
    )
    if len(baseline_points) < 2:
        line_id = line.get("id", "<unknown>")
        raise ValueError(f"TextLine {line_id!r} in {source} has no usable geometry")

    rect = cv2.minAreaRect(np.asarray(baseline_points, dtype=np.float32))
    center, (width, height), angle = rect
    if width <= 0 and height <= 0:
        raise ValueError(f"Degenerate baseline in {source}: {baseline_points!r}")
    if width < height:
        width = max(width, line_height)
    else:
        height = max(height, line_height)
    return cv2.boxPoints((center, (width, height), angle))


def _safe_xml(path: Path) -> etree._ElementTree:
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw.upper():
        raise ValueError(f"PAGE XML with DOCTYPE is not accepted: {path}")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    return etree.ElementTree(etree.fromstring(raw, parser=parser))


def _resolve_image(case: ManifestCase, tree: etree._ElementTree) -> Path:
    if case.image and case.image.is_file():
        return case.image
    if case.xml_path is None:
        raise ValueError(f"No PAGE XML path for case {case.id!r}")
    page = next(
        (
            element
            for element in tree.getroot().iter()
            if etree.QName(element).localname == "Page"
        ),
        None,
    )
    if page is None or not page.get("imageFilename"):
        raise ValueError(f"No image path for PAGE XML {case.xml_path}")
    image = (case.xml_path.parent / page.get("imageFilename")).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found for {case.xml_path}: {image}")
    return image


def _stage_case(
    case: ManifestCase,
    partition: str,
    index: int,
    root: Path,
    *,
    line_height: float,
) -> tuple[str, str, int, dict[str, str]]:
    if not case.xml_path or not case.xml_path.is_file():
        raise FileNotFoundError(
            f"PAGE XML not found for case {case.id!r}: {case.xml_path}"
        )
    tree = _safe_xml(case.xml_path)
    image = _resolve_image(case, tree)
    with Image.open(image) as opened:
        width, height = opened.size
        opened.verify()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {image}: {width}x{height}")

    stem = f"{index:05d}"
    image_name = stem + image.suffix.lower()
    image_rel = Path("images") / partition / image_name
    xml_rel = Path("xml") / partition / f"{stem}.xml"
    label_rel = Path("labels") / partition / f"{stem}.txt"
    for relative in (image_rel, xml_rel, label_rel):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, root / image_rel)

    page = next(
        (
            element
            for element in tree.getroot().iter()
            if etree.QName(element).localname == "Page"
        ),
        None,
    )
    if page is None:
        raise ValueError(f"No Page element in {case.xml_path}")
    page.set("imageFilename", f"../../images/{partition}/{image_name}")
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))

    labels: list[str] = []
    lines = [
        element
        for element in tree.getroot().iter()
        if etree.QName(element).localname == "TextLine"
    ]
    if not lines:
        raise ValueError(f"No TextLine geometry in {case.xml_path}")
    for line in lines:
        box = _line_box(line, case.xml_path, line_height)
        box[:, 0] = np.clip(box[:, 0], 0, width)
        box[:, 1] = np.clip(box[:, 1], 0, height)
        if abs(float(cv2.contourArea(box))) < 1:
            raise ValueError(f"Degenerate TextLine box in {case.xml_path}")

        coords = _find_child(line, "Coords")
        if coords is None:
            namespace = etree.QName(line).namespace
            tag = f"{{{namespace}}}Coords" if namespace else "Coords"
            coords = etree.Element(tag)
            line.insert(0, coords)
        if len(_points(coords.get("points"), case.xml_path)) < 3:
            coords.set("points", " ".join(f"{x:.0f},{y:.0f}" for x, y in box))
        normalized = [
            coordinate for x, y in box for coordinate in (x / width, y / height)
        ]
        labels.append("0 " + " ".join(f"{value:.8f}" for value in normalized))

    tree.write(root / xml_rel, encoding="utf-8", xml_declaration=True)
    (root / label_rel).write_text("\n".join(labels) + "\n", encoding="utf-8")
    return (
        f"{_REMOTE_ROOT}/{xml_rel.as_posix()}",
        f"{_REMOTE_ROOT}/{image_rel.as_posix()}",
        len(lines),
        {"xml_sha256": _sha256(case.xml_path), "image_sha256": _sha256(image)},
    )


def _selected_cases(
    manifest: Any, style_group: str | None, partition: str
) -> list[ManifestCase]:
    if style_group:
        return list(iter_style_group_cases(manifest, style_group, partition=partition))
    return manifest.get_partition(partition)


def prepare_segmenter_job(
    *,
    manifest_path: str | Path,
    backend: str,
    work_dir: str | Path,
    style_group: str | None = None,
    base_model_path: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 8,
    image_size: int = 1280,
    workers: int = 8,
    line_height: float = 32.0,
    augment: bool = True,
) -> SegmenterJob:
    """Validate data and build a deterministic, uploadable training job."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported segmenter backend {backend!r}; choose from {BACKENDS}"
        )
    if min(epochs, batch_size, image_size) <= 0 or workers < 0 or line_height <= 0:
        raise ValueError(
            "epochs, batch size, image size, and line height must be positive"
        )

    manifest = load_frozen_manifest(manifest_path)
    train_cases = _selected_cases(manifest, style_group, "train")
    validation_cases = _selected_cases(manifest, style_group, "validation")
    if not train_cases or not validation_cases:
        raise ValueError(
            "Segmenter training requires non-empty train and validation partitions"
        )

    base_model = Path(base_model_path).resolve() if base_model_path else None
    if backend in {"orli", "yolo-obb"} and base_model is None:
        raise ValueError(f"{backend} training requires --base-model")
    if base_model is not None and not base_model.is_file():
        raise FileNotFoundError(f"Base segmenter model not found: {base_model}")
    if backend == "yolo-obb" and base_model and base_model.suffix.lower() != ".pt":
        raise ValueError("YOLO-OBB base model must be a .pt checkpoint")

    work = Path(work_dir)
    root = work / "job"
    root.mkdir(parents=True, exist_ok=False)
    train_xml: list[str] = []
    val_xml: list[str] = []
    train_images: list[str] = []
    val_images: list[str] = []
    sources: list[dict[str, Any]] = []
    line_counts = {"train": 0, "validation": 0}
    for partition, cases, xml_paths, image_paths in (
        ("train", train_cases, train_xml, train_images),
        ("validation", validation_cases, val_xml, val_images),
    ):
        for index, case in enumerate(cases, start=1):
            xml_path, image_path, count, hashes = _stage_case(
                case, partition, index, root, line_height=line_height
            )
            xml_paths.append(xml_path)
            image_paths.append(image_path)
            line_counts[partition] += count
            sources.append(
                {
                    "partition": partition,
                    "case_id": case.id,
                    "manuscript_id": case.manuscript_id,
                    **hashes,
                }
            )

    train_inputs = {
        value
        for source in sources
        if source["partition"] == "train"
        for value in (source["xml_sha256"], source["image_sha256"])
    }
    validation_inputs = {
        value
        for source in sources
        if source["partition"] == "validation"
        for value in (source["xml_sha256"], source["image_sha256"])
    }
    if overlap := train_inputs & validation_inputs:
        raise ValueError(
            "Train/validation leakage: identical PAGE XML or image content appears in both partitions "
            f"({len(overlap)} shared file hash(es))"
        )

    (root / "train.txt").write_text("\n".join(train_xml) + "\n", encoding="utf-8")
    (root / "validation.txt").write_text("\n".join(val_xml) + "\n", encoding="utf-8")
    (root / "train_images.txt").write_text(
        "\n".join(train_images) + "\n", encoding="utf-8"
    )
    (root / "validation_images.txt").write_text(
        "\n".join(val_images) + "\n", encoding="utf-8"
    )

    remote_base = None
    if base_model:
        remote_base = f"{_REMOTE_ROOT}/base{base_model.suffix.lower()}"
        shutil.copyfile(base_model, root / Path(remote_base).name)

    output_dir = "/workspace/models/segmenter"
    common = ["--epochs", str(epochs)]
    setup_cmds = [
        "tar -xzf /workspace/segmenter-job.tar.gz -C /workspace",
        _SETUP[backend],
    ]
    artifact_pattern = "best_*.safetensors"
    artifact_path = None

    if backend == "blla":
        train_cmd = [
            "ketos",
            "-d",
            "auto",
            "--workers",
            str(workers),
            "segtrain",
            "--output",
            output_dir,
            "--training-data",
            f"{_REMOTE_ROOT}/train.txt",
            "--evaluation-data",
            f"{_REMOTE_ROOT}/validation.txt",
            "--format-type",
            "page",
            *common,
        ]
        if remote_base:
            train_cmd += ["--load", remote_base, "--resize", "union"]
        train_cmd.append("--augment" if augment else "--no-augment")
    elif backend == "orli":
        assert remote_base is not None
        setup_cmds += [
            f"orli compile --output {_REMOTE_ROOT}/train.arrow --allow-textless --resize 1920 1440 --files {_REMOTE_ROOT}/train.txt",
            f"orli compile --output {_REMOTE_ROOT}/validation.arrow --allow-textless --resize 1920 1440 --files {_REMOTE_ROOT}/validation.txt",
        ]
        train_cmd = [
            "orli",
            "--device",
            "auto",
            "--precision",
            "bf16-mixed",
            "--workers",
            str(workers),
            "train",
            "--output",
            output_dir,
            "--training-files",
            f"{_REMOTE_ROOT}/train.arrow",
            "--evaluation-files",
            f"{_REMOTE_ROOT}/validation.arrow",
            "--image-size",
            "1920",
            "1440",
            "--batch-size",
            str(batch_size),
            "--load",
            remote_base,
            *common,
            "--augment" if augment else "--no-augment",
        ]
    elif backend == "dfine":
        dfine_config = {
            "precision": "bf16-mixed",
            "device": "auto",
            "num_workers": workers,
            "num_threads": 1,
            "train": {
                "training_data": [f"{_REMOTE_ROOT}/train.txt"],
                "evaluation_data": [f"{_REMOTE_ROOT}/validation.txt"],
                "checkpoint_path": output_dir,
                "weights_format": "safetensors",
                "line_class_mapping": [["*", 1], ["DefaultLine", 1]],
                "region_class_mapping": [],
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": [image_size, image_size],
                "augment": augment,
            },
        }
        (root / "dfine.yaml").write_text(
            yaml.safe_dump(dfine_config, sort_keys=False), encoding="utf-8"
        )
        train_cmd = ["dfine", "--config", f"{_REMOTE_ROOT}/dfine.yaml", "train"]
        if remote_base:
            train_cmd += ["--load", remote_base, "--resize", "union"]
    else:
        data = {
            "path": _REMOTE_ROOT,
            "train": "train_images.txt",
            "val": "validation_images.txt",
            "names": {0: "line"},
        }
        (root / "yolo.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        setup_cmds.append("yolo settings sync=False")
        train_cmd = [
            "yolo",
            "obb",
            "train",
            f"model={remote_base}",
            f"data={_REMOTE_ROOT}/yolo.yaml",
            f"epochs={epochs}",
            f"imgsz={image_size}",
            f"batch={batch_size}",
            "device=0",
            "project=/workspace/models",
            "name=segmenter",
            "exist_ok=True",
            f"workers={workers}",
            f"augment={str(augment)}",
            "plots=False",
        ]
        artifact_path = f"{output_dir}/weights/best.pt"
        artifact_pattern = "best.pt"

    archive = work / "segmenter-job.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname="job", recursive=True)

    name_bits = [manifest.manifest_id, style_group or "all", backend]
    name = re.sub(r"[^a-zA-Z0-9-]+", "-", "-".join(name_bits)).strip("-")[:63]
    provenance = {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": _sha256(manifest.path),
        "style_group": style_group,
        "backend": backend,
        "train_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "line_counts": line_counts,
        "line_height": line_height,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "workers": workers,
        "augment": augment,
        "base_model_sha256": _sha256(base_model) if base_model else None,
        "train_command": train_cmd,
        "setup_commands": setup_cmds,
        "sources": sources,
    }
    return SegmenterJob(
        backend=backend,
        name=name,
        archive=archive,
        train_cmd=train_cmd,
        setup_cmds=setup_cmds,
        artifact_remote_dir=output_dir,
        artifact_pattern=artifact_pattern,
        artifact_remote_path=artifact_path,
        provenance=provenance,
    )


def train_segmenter_remote(
    *,
    runner: RunPodRunner | None,
    output_model_path: str | Path,
    reports_dir: str | Path,
    dry_run: bool = False,
    **job_options: Any,
) -> dict[str, Any]:
    """Prepare one candidate job and optionally run its RunPod lifecycle."""
    output = Path(output_model_path)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        job = prepare_segmenter_job(work_dir=temp_dir, **job_options)
        report = dict(job.provenance)
        report["output_model"] = str(output)
        report["status"] = "dry-run" if dry_run else "completed"
        report_path = reports / f"{job.name}-training.json"
        if not dry_run:
            if runner is None:
                raise ValueError("runner is required unless dry_run is enabled")
            runner.run_training(
                name=job.name,
                train_cmd=job.train_cmd,
                artifact_remote_dir=job.artifact_remote_dir,
                artifact_remote_path=job.artifact_remote_path,
                artifact_pattern=job.artifact_pattern,
                artifact_local_path=str(output),
                pre_train_upload=[
                    (str(job.archive), "/workspace/segmenter-job.tar.gz")
                ],
                setup_cmds=job.setup_cmds,
            )
            report["output_model_sha256"] = _sha256(output)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["report_path"] = str(report_path)
        return report
