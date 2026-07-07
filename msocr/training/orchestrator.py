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

# ponytail: safetensors 0.7 refuses to load LSTM tied weights (weight_ih_l0
# shared internally by VGSL) when no name in the shared group has "complete"
# storage in the file. The converted sophro base hits this on ketos load.
# Workaround from https://huggingface.co/docs/safetensors/torch_shared_tensors:
# when complete_names is empty, fall back to the first name in the shared
# group instead of raising. Source-file edit on the pod (same pattern as the
# checkpoint patch). Idempotent via marker.
_SAFETENSORS_SHARED_TENSOR_PATCH = r"""python3 -c "
import safetensors.torch as st
p = st.__file__
s = open(p).read()
marker = '# msocr-shared-tensor-patch'
if marker not in s:
    old = '''        if not complete_names:
            raise RuntimeError(
                \"Error while trying to find names to remove to save state dict, but found no suitable name to keep\"
                f\" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model\"
                \" since you could be storing much more memory than needed. Please refer to\"
                \" https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an\"
                \" issue.\"
            )

        keep_name = sorted(list(complete_names))[0]'''
    new = '''        if not complete_names:
            # msocr-shared-tensor-patch: fall back to first name when no
            # complete storage exists (VGSL LSTM tied weights on load).
            keep_name = sorted(list(shared))[0]
        else:
            keep_name = sorted(list(complete_names))[0]'''
    assert old in s, 'safetensors _remove_duplicate_names shape changed'
    s = s.replace(old, new)
    open(p, 'w').write(s)
print('safetensors shared-tensor patch applied')
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

    tree = ET.parse(str(src_xml))
    root = tree.getroot()
    # ponytail: detect PAGE XML namespace from root tag (works for 2019 and 2013);
    # fall back to 2019 only if root has no namespace (should not normally happen).
    NS = root.tag.split("}")[0][1:] if root.tag.startswith("{") else "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
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
    tree = ET.parse(str(src_xml))
    root = tree.getroot()
    # ponytail: detect PAGE XML namespace from root tag (works for 2019 and 2013);
    # fall back to 2019 only if root has no namespace (should not normally happen).
    NS = root.tag.split("}")[0][1:] if root.tag.startswith("{") else "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    page = root.find(f"{{{NS}}}Page")
    if page is None:
        raise ValueError(f"No <Page> in {src_xml}")
    fname = page.get("imageFilename")
    if not fname:
        raise ValueError(f"No imageFilename in {src_xml} and no image hint provided")
    p = src_xml.parent / fname
    if not p.exists():
        # ponytail: case-insensitive fallback (MS Jer 36 XMLs may reference a
        # different extension case than the file on disk), plus a sibling
        # images/ directory lookup (Jer 36 stores images alongside page/, not
        # in the same dir). Strict path kept first so c2av/Vienna unchanged.
        target = fname.lower()
        dirs = [src_xml.parent, src_xml.parent.parent / "images"]
        match = None
        for d in dirs:
            if not d.is_dir():
                continue
            match = next((f for f in d.iterdir() if f.name.lower() == target), None)
            if match is not None:
                break
        if match is None:
            raise FileNotFoundError(f"Image not found for {src_xml}: {p}")
        p = match
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
    warmup: int = 0,  # ponytail: 0 = off; research recommends 200 for fine-tuning stability.
    lr: float | None = None,  # ponytail: None = ketos default (1e-3); 1e-4 recommended for small-data fine-tune.
    device: str = "auto",  # ponytail: ketos 7.0.2 crashes on `-d cuda`; auto lets pytorch pick the GPU.
    workers: int = 8,
    quit_mode: str = "fixed",
    setup_cmds: list[str] | None = None,
    freeze_old_rows: bool = False,
    codec_json_path: str | None = None,
) -> dict:
    """Train + evaluate one style-group. Returns the eval report dict.

    Base model resolution (first wins): explicit ``base_model_path`` arg >
    style_group ``base_model_override`` in manifest > ``DEFAULT_BASE_MODELS``
    for the manifest's ``script_block``. If none match, trains from scratch.
    If ``setup_cmds`` is None, defaults to installing kraken into the pod image.

    When ``freeze_old_rows`` is True, swaps the training command from
    `ketos train` to the §1 row-freeze Python-API harness
    (`msocr.training.ketos_trainer_api`). The harness file + ``codec_json_path``
    are uploaded to the pod and run via `python3`. Requires ``codec_json_path``
    to point to the Phase 0a union codec JSON (default: reports/c2av_union_codec.json).
    """
    if setup_cmds is None:
        setup_cmds = [
            "python3 -m pip install --quiet 'kraken>=7.0.2'",
            _KRAKEN_CHECKPOINT_PATCH,
            _SAFETENSORS_SHARED_TENSOR_PATCH,
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
    # ponytail: polygonization is ~1.1 min/folio CPU-bound serial — 128 folios
    # = ~2.5h. Cache enriched XMLs by (xml_path, mtime, size, target_img_name)
    # so re-runs skip the work. Cache lives under ~/.cache/msocr/poly_cache/.
    cache_dir = Path.home() / ".cache" / "msocr" / "poly_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
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
            remote_img_name = f"{part}_{idx}.png"
            # ponytail: cache key — source xml path + mtime + size + target image
            # name. If the source XML or image changes, the key changes and we
            # re-polygonize. Idempotent: a hit returns the cached enriched file.
            src_stat = c.xml_path.stat()
            import hashlib
            key_src = f"{c.xml_path}|{src_stat.st_mtime_ns}|{src_stat.st_size}|{remote_img_name}"
            key = hashlib.sha1(key_src.encode()).hexdigest()
            cached = cache_dir / f"{key}.xml"
            enriched_xml = tmp_path / f"{part}_{idx}_poly.xml"
            if cached.exists() and cached.stat().st_size > 0:
                # ponytail: copy not rename — cache is on ~/.cache (home fs)
                # but tmp is on a different filesystem (tmpfs/overlay), so
                # os.replace raises EXDEV. copy + small read is cheaper than
                # re-polygonizing ~1.1 min/folio.
                enriched_xml.write_bytes(cached.read_bytes())
            else:
                _enrich_xml_with_polygons(c.xml_path, image, enriched_xml, target_image_name=remote_img_name)
                # Write atomically: copy to a temp then rename, so a crash mid-
                # write never leaves a partial cache entry.
                tmp_cache = cached.with_suffix(".xml.tmp")
                tmp_cache.write_bytes(enriched_xml.read_bytes())
                tmp_cache.rename(cached)
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
        if freeze_old_rows:
            # ponytail: §1 row-freeze path. The ketos CLI has no per-row freeze
            # flag, so we ship the Python-API harness (Phase 0b) to the pod
            # and run it via `python3`. The harness uses KrakenTrainer directly
            # with a register_hook that zeros old-row grads every step.
            harness_src = Path(__file__).parent / "ketos_trainer_api.py"
            if not harness_src.exists():
                raise FileNotFoundError(f"Row-freeze harness not found: {harness_src}")
            codec_json = Path(codec_json_path) if codec_json_path else Path("reports/c2av_union_codec.json")
            if not codec_json.exists():
                raise FileNotFoundError(
                    f"Codec JSON not found: {codec_json}. Run `scripts/dump_codec.py` first (Phase 0a)."
                )
            pre_train_upload.append((str(harness_src), "/workspace/ketos_trainer_api.py"))
            pre_train_upload.append((str(codec_json), "/workspace/codec.json"))
            # ponytail: when freeze_old_rows is set, force freeze_backbone to
            # 999999 (whole run) — the §1 mechanism assumes a frozen backbone.
            effective_freeze_backbone = 999999
            train_cmd = [
                "python3", "/workspace/ketos_trainer_api.py",
                "--load", "/workspace/base.safetensors" if load_model_path else "scratch",
                "--train-manifest", "/workspace/train_manifest.txt",
                "--eval-manifest", "/workspace/val_manifest.txt",
                "--codec-json", "/workspace/codec.json",
                "--checkpoint-path", "/workspace/models/" + style_group_id,
                "--format-type", "page",
                "--resize", "union",
                "--epochs", str(epochs),
                "--min-epochs", str(min_epochs),
                "--lag", str(lag),
                "--lrate", str(lr if lr is not None else 1e-4),
                "--warmup", str(warmup),
                "--quit", quit_mode,
                "--freeze-backbone", str(effective_freeze_backbone),
                "--num-workers", str(workers),
                "--accelerator", device if device != "auto" else "auto",
            ]
            if augment:
                train_cmd.append("--augment")
        else:
            train_cmd = [
                "ketos", "-d", device, "--workers", str(workers), "train",
                "--quit", quit_mode,
                "--epochs", str(epochs),
                "--min-epochs", str(min_epochs),
                "--lag", str(lag),
                "-f", "page",
                "-t", "/workspace/train_manifest.txt",
                "-e", "/workspace/val_manifest.txt",
                "-o", "/workspace/models/" + style_group_id,
            ]
            if load_model_path:
                train_cmd += [
                    "--load", "/workspace/base.safetensors",
                    # ponytail: kraken 7.0.2 vgsl.py setup() only dispatches `fail`,
                    # `union`, or `new` — the `add`/`both` advertised by ketos --help are
                    # argparse-only and raise ValueError at runtime. `union` preserves
                    # the base codec, appends unseen codepoints, and resizes only the
                    # output layer — exactly the Fix A transfer-learning scenario
                    # (22 shared Syriac consonants keep their weights; 5 new Sogdian
                    # classes learn from scratch). `new` rebuilds the codec from
                    # scratch and was the cause of runs #1/#2 failing.
                    "--resize", "union",
                    "--freeze-backbone", str(freeze_backbone),
                ]
            if augment:
                train_cmd.append("--augment")
            if warmup > 0:
                train_cmd += ["--warmup", str(warmup)]
            if lr is not None:
                train_cmd += ["-r", str(lr)]

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
