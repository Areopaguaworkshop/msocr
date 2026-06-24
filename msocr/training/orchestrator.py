"""Procedural per-style-group training orchestrator.

Per design D8 (Approach A): walk a style_group in a manifest,
enrich each PAGE XML with polygon <Coords> (kraken 7.x requires them;
our annotation tool only exports <Baseline>), upload XML+image to a
RunPod pod, run training there, download the .safetensors artifact,
run evaluation locally on the downloaded model. One style-group at a
time. No queue, no DAG.

Ponytail: if we need durable parallelism later, wrap this in RQ.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from msocr.data.manifest import load_frozen_manifest, iter_style_group_cases
from msocr.training.runpod_runner import RunPodRunner
from msocr.evaluation.harness import run_evaluation

# kraken 7.0.2 crashes on checkpoint save when training from scratch
# (self.net is None until setup()). One-line guard in vgsl.py + base.py.
# Applied on the pod after pip install. Idempotent: skips if already patched.
# Upstream bug, unreported as of 2026-06.
_KRAKEN_CHECKPOINT_PATCH = r"""python3 -c "
import kraken.train.vgsl as v, kraken.train.base as b
for p, old, new, marker in [
    (v.__file__, 'self.hparams.config.spec = self.net.spec', 'if self.net is not None: self.hparams.config.spec = self.net.spec', 'if self.net is not None: self.hparams.config.spec = self.net.spec'),
    (b.__file__, 'if metrics:\n            self.net.user_metadata', 'if metrics and self.net is not None:\n            self.net.user_metadata', 'if metrics and self.net is not None:\n            self.net.user_metadata'),
]:
    s = open(p).read()
    if marker not in s:
        s = s.replace(old, new)
        open(p, 'w').write(s)
"
"""


def _enrich_xml_with_polygons(src_xml: Path, image: Path, out_xml: Path, *, target_image_name: str | None = None) -> Path:
    """Compute <Coords> for each <TextLine> and write a new PAGE XML.

    kraken 7.x requires <Coords> per <TextLine>; our export only has
    <Baseline>. Falls back to the image path next to the XML if not given.

    If ``target_image_name`` is set, rewrite <Page imageFilename> so the XML
    resolves the image by its uploaded basename (kraken opens imageFilename
    relative to the XML, not the original filesystem path).
    """
    import lxml.etree as ET
    from PIL import Image
    from kraken.lib.segmentation import calculate_polygonal_environment

    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    tree = ET.parse(str(src_xml))
    root = tree.getroot()
    page = root.find(f"{{{NS}}}Page")
    if page is None:
        raise ValueError(f"No <Page> in {src_xml}")
    if target_image_name:
        page.set("imageFilename", target_image_name)
    im = Image.open(str(image)).convert("L")
    baselines: list[list[tuple[int, int]]] = []
    lines_xml: list[ET._Element] = []
    for tl in page.iter(f"{{{NS}}}TextLine"):
        bl = tl.find(f"{{{NS}}}Baseline")
        if bl is None:
            continue
        pts = [(int(x), int(y)) for x, y in (p.split(",") for p in bl.get("points").split())]
        baselines.append(pts)
        lines_xml.append(tl)
    polys = calculate_polygonal_environment(im=im, baselines=baselines, topline=False, raise_on_error=False)
    for tl, poly in zip(lines_xml, polys):
        if poly is None:
            continue
        pts_str = " ".join(f"{int(x)},{int(y)}" for x, y in poly)
        ET.SubElement(tl, f"{{{NS}}}Coords", {"points": pts_str})
    tree.write(str(out_xml), xml_declaration=True, encoding="utf-8")
    return out_xml


def _resolve_image_for_xml(src_xml: Path, hinted_image: Path | None) -> Path:
    """Find the image for a PAGE XML: explicit hint, or imageFilename next to XML."""
    if hinted_image and hinted_image.exists():
        return hinted_image
    import lxml.etree as ET
    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    tree = ET.parse(str(src_xml))
    page = tree.getroot().find(f"{{{NS}}}Page")
    if page is None:
        raise ValueError(f"No <Page> in {src_xml}")
    fname = page.get("imageFilename")
    if not fname:
        raise ValueError(f"No imageFilename in {src_xml} and no image hint provided")
    p = src_xml.parent / fname
    if not p.exists():
        raise FileNotFoundError(f"Image not found for {src_xml}: {p}")
    return p


def walk_style_group(
    manifest_path: str,
    style_group_id: str,
    runner: RunPodRunner,
    output_model_path: str,
    reports_dir: str,
    base_model_path: str | None = None,
    epochs: int = 50,
    min_epochs: int = 20,
    lag: int = 10,
    freeze_backbone: int = 0,
    augment: bool = True,
    device: str = "auto",  # ponytail: ketos 7.0.2 crashes on `-d cuda`; auto lets pytorch pick the GPU.
    workers: int = 8,
    quit_mode: str = "fixed",
    setup_cmds: list[str] | None = None,
) -> dict:
    """Train + evaluate one style-group. Returns the eval report dict.

    Base model resolution (first wins): explicit ``base_model_path`` arg >
    style_group ``base_model_override`` in manifest > ``DEFAULT_BASE_MODELS``
    for the manifest's ``script_block``. If none match, trains from scratch.
    If ``setup_cmds`` is None, defaults to installing kraken into the pod image.
    """
    if setup_cmds is None:
        setup_cmds = [
            "python3 -m pip install --quiet 'kraken>=7.0.2'",
            _KRAKEN_CHECKPOINT_PATCH,
        ]
    manifest = load_frozen_manifest(manifest_path)
    sg = (manifest.style_groups or {}).get(style_group_id) or {}
    base_override = sg.get("base_model_override")
    # ponytail: resolve base model in priority order — explicit arg > style_group
    # override > manifest script_block default. None of these = train from scratch.
    if base_override:
        load_model = base_override
    elif base_model_path:
        load_model = base_model_path
    else:
        from msocr.language_registry import default_base_model_for_script_block
        default = default_base_model_for_script_block(manifest.script_block)
        load_model = str(default) if default else None
    load_model_path = Path(load_model) if load_model else None
    if load_model_path and not load_model_path.exists():
        raise FileNotFoundError(f"Base model for RunPod training not found: {load_model_path}")

    train_cases = list(iter_style_group_cases(manifest, style_group_id, partition="train"))
    val_cases = list(iter_style_group_cases(manifest, style_group_id, partition="validation"))
    if not train_cases:
        raise ValueError(f"No train cases in style_group {style_group_id!r}")

    # Enrich each XML with polygon <Coords> (kraken 7.x requires them),
    # then upload XML + image to the pod. Train/val manifests list paths.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_manifest_lines: list[str] = []
        val_manifest_lines: list[str] = []
        pre_train_upload: list[tuple[str, str]] = []
        all_cases = [("train", c, train_manifest_lines) for c in train_cases] + \
                    [("val", c, val_manifest_lines) for c in val_cases]
        for idx, (part, c, manifest_lines) in enumerate(all_cases):
            if not c.xml_path:
                continue
            image = _resolve_image_for_xml(c.xml_path, c.image)
            enriched_xml = tmp_path / f"{part}_{idx}_poly.xml"
            remote_img_name = f"{part}_{idx}.png"
            _enrich_xml_with_polygons(c.xml_path, image, enriched_xml, target_image_name=remote_img_name)
            remote_xml = f"/workspace/{part}_{idx}.xml"
            remote_img = f"/workspace/{remote_img_name}"
            pre_train_upload.append((str(enriched_xml), remote_xml))
            pre_train_upload.append((str(image), remote_img))
            manifest_lines.append(remote_xml)

        train_manifest = tmp_path / "train_manifest.txt"
        val_manifest = tmp_path / "val_manifest.txt"
        train_manifest.write_text("\n".join(train_manifest_lines))
        val_manifest.write_text("\n".join(val_manifest_lines))
        pre_train_upload.append((str(train_manifest), "/workspace/train_manifest.txt"))
        pre_train_upload.append((str(val_manifest), "/workspace/val_manifest.txt"))
        if load_model_path:
            pre_train_upload.append((str(load_model_path), "/workspace/base.safetensors"))

        # ketos 7.0: -t/-e expect text manifests (one path per line);
        # -f page parses positional XML args. We pass XML via manifests.
        train_cmd = [
            "ketos", "-d", device, "--workers", str(workers), "train",
            "--quit", quit_mode,
            "--epochs", str(epochs),
            "-f", "page",
            "-t", "/workspace/train_manifest.txt",
            "-e", "/workspace/val_manifest.txt",
            "-o", "/workspace/models/" + style_group_id,
        ]
        if load_model_path:
            train_cmd += [
                "--load", "/workspace/base.safetensors",
                "--resize", "new",
                "--freeze-backbone", str(freeze_backbone),
            ]
        if augment:
            train_cmd.append("--augment")

        runner.run_training(
            name=f"{manifest.manifest_id}-{style_group_id}",
            train_cmd=train_cmd,
            # ponytail: ketos 7.0 writes best_{score:.4f}.safetensors into the -o dir;
            # score is unknown until training ends, so we glob the dir post-train.
            # Runner does the glob via ssh_exec + download_artifact against the resolved name.
            artifact_remote_dir=f"/workspace/models/{style_group_id}",
            artifact_local_path=output_model_path,
            pre_train_upload=pre_train_upload,
            setup_cmds=setup_cmds,
        )

    return run_evaluation(
        manifest_path=manifest_path,
        style_group_id=style_group_id,
        model_path=output_model_path,
        reports_dir=reports_dir,
    )


def _dataset_cfg(prefix: Path) -> dict:
    """Deprecated: kept only for backwards compat. No longer used by walk_style_group."""
    return {
        "dataset": {"format_type": "xml"},
        "model": {"spec": "placeholder"},
        "training": {"epochs": 0, "device": "cpu", "workers": 1},
        "output": {"model_prefix": str(prefix.with_suffix(""))},
    }
