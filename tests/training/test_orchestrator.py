"""Tests for msocr.training.orchestrator. Runner, eval, and polygon
enrichment are mocked — we only verify the orchestrator's command/upload shape.
"""
import json
from unittest.mock import patch, MagicMock

from msocr.training.orchestrator import walk_style_group


def _write_page_xml(path, image_filename, n_lines=1):
    """Minimal PAGE XML with n_lines Baselines (no Coords — enrichment adds them)."""
    lines = "".join(
        f'<TextLine id="l{i}"><Baseline points="10,{20+i*30} 100,{20+i*30}" />'
        f'<TextEquiv><Unicode>line{i}</Unicode></TextEquiv></TextLine>'
        for i in range(n_lines)
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="{image_filename}" imageWidth="200" imageHeight="200">'
        f'<TextRegion id="r1"><Coords points="0,0 200,0 200,200 0,200" />{lines}</TextRegion>'
        f'</Page></PcGts>'
    )
    path.write_text(xml)


def _make_manifest(tmp_path, style_groups=None, n_train=1, n_val=1):
    """Build a fixture manifest on disk with real tiny PAGE XML + image files."""
    items = []
    for i in range(n_train + n_val):
        img = tmp_path / f"img{i}.png"
        img.write_bytes(b"fake-png")  # enrichment/PIL is mocked, bytes irrelevant
        xml = tmp_path / f"img{i}.xml"
        _write_page_xml(xml, f"img{i}.png", n_lines=1)
        items.append((xml, img))
    train_items = [{"id": f"t{i}", "manuscript_id": "M1", "image": str(items[i][1]), "xml_path": str(items[i][0])}
                   for i in range(n_train)]
    val_items = [{"id": f"v{i}", "manuscript_id": "M2", "image": str(items[n_train + i][1]), "xml_path": str(items[n_train + i][0])}
                 for i in range(n_val)]
    holdout_items = [{"id": "h0", "manuscript_id": "M3", "image": str(items[0][1]), "xml_path": str(items[0][0])}]
    manifest = {
        "manifest_id": "test-v1",
        "writing_mode": "handwritten",
        "language": "sogdian",
        "script_block": "U+10F30",
        "partitions": {"train": train_items, "validation": val_items, "holdout": holdout_items},
        "style_groups": style_groups or {"g1": {"manuscript_ids": ["M1", "M2", "M3"]}},
    }
    p = tmp_path / "test-v1.json"
    p.write_text(json.dumps(manifest))
    return p


def _mock_enrich(src_xml, image, out_xml, *, target_image_name=None):
    """Bypass PIL/kraken — just copy the src XML to out_xml."""
    from pathlib import Path
    import lxml.etree as ET

    if target_image_name is None:
        Path(out_xml).write_bytes(Path(src_xml).read_bytes())
        return out_xml

    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    tree = ET.parse(str(src_xml))
    page = tree.getroot().find(f"{{{ns}}}Page")
    page.set("imageFilename", target_image_name)
    tree.write(str(out_xml), xml_declaration=True, encoding="utf-8")
    return out_xml


def test_walk_style_group_builds_train_cmd_and_runs_eval(tmp_path):
    """With a base model: --load/--resize/--freeze-backbone emitted; uploads XML+image+manifests."""
    manifest_path = _make_manifest(tmp_path)
    base_model = tmp_path / "base.safetensors"
    base_model.write_bytes(b"base")

    fake_runner = MagicMock()
    def _assert_uploaded_xml_rewrites_image(**kwargs):
        upload_by_remote = {remote: local for local, remote in kwargs["pre_train_upload"]}
        train_xml = upload_by_remote["/workspace/train_0.xml"]
        assert 'imageFilename="train_0.png"' in open(train_xml).read()
        return "/tmp/out.safetensors"

    fake_runner.run_training.side_effect = _assert_uploaded_xml_rewrites_image
    fake_report = {"per_style_group": {"g1": {"cer": 0.05}}, "per_manuscript": {}}

    with patch("msocr.training.orchestrator._enrich_xml_with_polygons", _mock_enrich), \
         patch("msocr.training.orchestrator.run_evaluation", return_value=fake_report) as fake_eval:
        report = walk_style_group(
            manifest_path=str(manifest_path),
            style_group_id="g1",
            runner=fake_runner,
            base_model_path=str(base_model),
            output_model_path="/tmp/out.safetensors",
            reports_dir="/tmp/reports",
            augment=True,
            device="cuda:0",
        )

    fake_runner.run_training.assert_called_once()
    _, kwargs = fake_runner.run_training.call_args
    train_cmd = kwargs["train_cmd"]
    assert train_cmd[:6] == ["ketos", "-d", "cuda:0", "--workers", "8", "train"]
    assert "-f" in train_cmd and "page" in train_cmd
    assert "-t" in train_cmd and "/workspace/train_manifest.txt" in train_cmd
    assert "-e" in train_cmd and "/workspace/val_manifest.txt" in train_cmd
    assert "--load" in train_cmd and "--resize" in train_cmd and "--freeze-backbone" in train_cmd
    assert "--augment" in train_cmd and train_cmd.count("--augment") == 1
    assert "--quit" in train_cmd and "fixed" in train_cmd

    uploads = kwargs["pre_train_upload"]
    # 1 train XML + 1 train image + 1 val XML + 1 val image + 2 manifests + 1 base = 7
    assert len(uploads) == 7
    remote_paths = [dst for _, dst in uploads]
    assert "/workspace/train_manifest.txt" in remote_paths
    assert "/workspace/val_manifest.txt" in remote_paths
    assert "/workspace/base.safetensors" in remote_paths
    assert any(p.endswith(".xml") for p in remote_paths)
    assert any(p.endswith(".png") for p in remote_paths)

    # setup_cmds defaults to kraken-install + checkpoint patch.
    setup = kwargs["setup_cmds"]
    assert len(setup) == 2
    assert "kraken" in setup[0]
    assert "self.net is not None" in setup[1]

    fake_eval.assert_called_once()
    _, eval_kwargs = fake_eval.call_args
    assert eval_kwargs["model_path"] == "/tmp/out.safetensors"
    assert eval_kwargs["style_group_id"] == "g1"
    assert report == fake_report


def test_walk_style_group_from_scratch_no_load(tmp_path):
    """With no base model resolvable, no --load/--resize/--freeze-backbone, no base in uploads."""
    manifest_path = _make_manifest(tmp_path)

    fake_runner = MagicMock()
    fake_runner.run_training.return_value = "/tmp/out.safetensors"

    # ponytail: script_block U+10F30 now resolves to a default base model; clear
    # the registry to force the from-scratch path and test it in isolation.
    with patch("msocr.language_registry.DEFAULT_BASE_MODELS", {}), \
         patch("msocr.training.orchestrator._enrich_xml_with_polygons", _mock_enrich), \
         patch("msocr.training.orchestrator.run_evaluation", return_value={}) as fake_eval:
        walk_style_group(
            manifest_path=str(manifest_path),
            style_group_id="g1",
            runner=fake_runner,
            base_model_path=None,
            output_model_path="/tmp/out.safetensors",
            reports_dir="/tmp/reports",
            device="cuda:0",
        )

    _, kwargs = fake_runner.run_training.call_args
    cmd = kwargs["train_cmd"]
    assert "--load" not in cmd
    assert "--resize" not in cmd
    assert "--freeze-backbone" not in cmd
    uploads = kwargs["pre_train_upload"]
    assert all("/workspace/base.safetensors" not in dst for _, dst in uploads)


def test_walk_style_group_respects_no_augment(tmp_path):
    """--no-augment should remove --augment from the pod-side ketos command."""
    manifest_path = _make_manifest(tmp_path)
    base_model = tmp_path / "base.safetensors"
    base_model.write_bytes(b"base")
    fake_runner = MagicMock()

    with patch("msocr.training.orchestrator._enrich_xml_with_polygons", _mock_enrich), \
         patch("msocr.training.orchestrator.run_evaluation", return_value={}):
        walk_style_group(
            manifest_path=str(manifest_path),
            style_group_id="g1",
            runner=fake_runner,
            base_model_path=str(base_model),
            output_model_path="/tmp/out.safetensors",
            reports_dir="/tmp/reports",
            augment=False,
        )

    _, kwargs = fake_runner.run_training.call_args
    assert "--augment" not in kwargs["train_cmd"]


def test_walk_style_group_resolves_script_block_default_base(tmp_path):
    """No explicit base_model_path + script_block U+10F30 → DEFAULT_BASE_MODELS[...] is loaded."""
    manifest_path = _make_manifest(tmp_path)
    avestan = tmp_path / "avestan.mlmodel"
    avestan.write_bytes(b"avestan")

    fake_runner = MagicMock()
    fake_runner.run_training.return_value = "/tmp/out.safetensors"
    with patch("msocr.language_registry.DEFAULT_BASE_MODELS",
               {"U+10F30": str(avestan)}), \
         patch("msocr.training.orchestrator._enrich_xml_with_polygons", _mock_enrich), \
         patch("msocr.training.orchestrator.run_evaluation", return_value={}):
        walk_style_group(
            manifest_path=str(manifest_path),
            style_group_id="g1",
            runner=fake_runner,
            base_model_path=None,
            output_model_path="/tmp/out.safetensors",
            reports_dir="/tmp/reports",
            device="cuda:0",
        )

    _, kwargs = fake_runner.run_training.call_args
    cmd = kwargs["train_cmd"]
    assert "--load" in cmd and "--resize" in cmd
    uploads = kwargs["pre_train_upload"]
    assert any(dst == "/workspace/base.safetensors" for _, dst in uploads)
