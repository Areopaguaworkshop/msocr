"""Checks for the shared four-backend segmenter training path."""

import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from msocr.training.segmenter_trainer import (
    prepare_segmenter_job,
    train_segmenter_remote,
)


def _page(path: Path, image_name: str, baseline: str = "10,40 90,42") -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="{image_name}" imageWidth="100" imageHeight="80">'
        '<TextRegion id="r1"><Coords points="0,0 99,0 99,79 0,79" />'
        f'<TextLine id="l1"><Baseline points="{baseline}" /></TextLine>'
        "</TextRegion></Page></PcGts>",
        encoding="utf-8",
    )


def _manifest(tmp_path: Path, *, duplicate: bool = False) -> Path:
    train_image = tmp_path / "train.png"
    val_image = train_image if duplicate else tmp_path / "val.png"
    Image.new("RGB", (100, 80), "white").save(train_image)
    if not duplicate:
        Image.new("RGB", (100, 80), "gray").save(val_image)
    train_xml = tmp_path / "train.xml"
    val_xml = train_xml if duplicate else tmp_path / "val.xml"
    _page(train_xml, train_image.name)
    if not duplicate:
        _page(val_xml, val_image.name, baseline="8,35 92,37")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_id": "layout-test",
                "writing_mode": "handwritten",
                "language": "sogdian",
                "script_block": "U+0710",
                "partitions": {
                    "train": [
                        {
                            "id": "train",
                            "manuscript_id": "train-ms",
                            "xml_path": str(train_xml),
                            "image": str(train_image),
                        }
                    ],
                    "validation": [
                        {
                            "id": "val",
                            "manuscript_id": "val-ms",
                            "xml_path": str(val_xml),
                            "image": str(val_image),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize("backend", ["blla", "orli", "dfine", "yolo-obb"])
def test_prepare_segmenter_job_builds_each_backend(tmp_path, backend):
    base = None
    if backend in {"orli", "yolo-obb"}:
        base = tmp_path / ("base.pt" if backend == "yolo-obb" else "base.safetensors")
        base.write_bytes(b"test weights")

    work = tmp_path / backend
    work.mkdir()
    job = prepare_segmenter_job(
        manifest_path=_manifest(tmp_path),
        backend=backend,
        base_model_path=base,
        work_dir=work,
        epochs=2,
    )

    assert job.archive.is_file()
    assert job.provenance["line_counts"] == {"train": 1, "validation": 1}
    assert job.train_cmd[0] in {"ketos", "orli", "dfine", "yolo"}
    with tarfile.open(job.archive) as archive:
        names = archive.getnames()
        assert "job/train.txt" in names
        assert "job/labels/train/00001.txt" in names
        label = archive.extractfile("job/labels/train/00001.txt").read().decode()
    values = [float(value) for value in label.split()[1:]]
    assert len(values) == 8
    assert all(0 <= value <= 1 for value in values)


def test_prepare_segmenter_job_rejects_cross_partition_file_leakage(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="Train/validation leakage"):
        prepare_segmenter_job(
            manifest_path=_manifest(tmp_path, duplicate=True),
            backend="blla",
            work_dir=work,
        )


def test_prepare_segmenter_job_rejects_xml_doctype(tmp_path):
    manifest = _manifest(tmp_path)
    train_xml = tmp_path / "train.xml"
    train_xml.write_text(
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="DOCTYPE"):
        prepare_segmenter_job(
            manifest_path=manifest,
            backend="blla",
            work_dir=work,
        )


def test_prepare_segmenter_job_repairs_unusable_coords_from_baseline(tmp_path):
    manifest = _manifest(tmp_path)
    train_xml = tmp_path / "train.xml"
    train_xml.write_text(
        train_xml.read_text().replace(
            '<TextLine id="l1">', '<TextLine id="l1"><Coords points="10,40" />'
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()

    job = prepare_segmenter_job(
        manifest_path=manifest,
        backend="blla",
        work_dir=work,
    )

    with tarfile.open(job.archive) as archive:
        xml = archive.extractfile("job/xml/train/00001.xml").read().decode()
    assert 'points="10,40"' not in xml


def test_train_segmenter_remote_uses_one_runner_lifecycle(tmp_path):
    class FakeRunner:
        def __init__(self):
            self.kwargs = None

        def run_training(self, **kwargs):
            self.kwargs = kwargs
            Path(kwargs["artifact_local_path"]).write_bytes(b"trained")

    runner = FakeRunner()
    output = tmp_path / "model.safetensors"
    report = train_segmenter_remote(
        runner=runner,
        output_model_path=output,
        reports_dir=tmp_path / "reports",
        manifest_path=_manifest(tmp_path),
        backend="blla",
        epochs=1,
    )

    assert output.read_bytes() == b"trained"
    assert report["status"] == "completed"
    assert runner.kwargs["pre_train_upload"][0][1] == "/workspace/segmenter-job.tar.gz"
    assert Path(report["report_path"]).is_file()
