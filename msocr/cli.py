"""CLI interface for Sogdian manuscript HTR."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import click
import yaml

from msocr.data.manifest import load_frozen_manifest
from msocr.language_registry import (
    CLI_LANGUAGE_ALIASES,
    CLI_LANGUAGE_CODES,
    normalize_language_code,
)
from msocr.training.ketos_trainer import KetosTrainer


LANG_CHOICES = click.Choice([*CLI_LANGUAGE_CODES, *CLI_LANGUAGE_ALIASES], case_sensitive=False)
OUTPUT_FORMAT_CHOICES = click.Choice(["json", "markdown"], case_sensitive=False)
ENGINE_NAME = "kraken"
DEFAULT_CONFIGS: Dict[str, str] = {
    "sogdian": "msocr/configs/sogdian_config.yaml",
}


def _normalize_lang(lang: str) -> str:
    try:
        return normalize_language_code(lang)
    except KeyError:
        return lang.strip().lower()


def _resolve_default_config(lang: str) -> Path:
    lang_key = _normalize_lang(lang)
    config_path = DEFAULT_CONFIGS.get(lang_key)
    if not config_path:
        raise click.ClickException(
            f"No default HTR config for lang={lang_key}. Pass --config explicitly."
        )
    resolved = Path(config_path)
    if not resolved.exists():
        raise click.ClickException(f"Default config does not exist: {resolved}")
    return resolved


def _collect_xml_files(gt_dir: Path | None, gt_file: str | None) -> List[Path]:
    xml_files: List[Path] = []
    if gt_file:
        xml_path = Path(gt_file)
        if not xml_path.exists():
            raise click.ClickException(f"Ground-truth XML file not found: {xml_path}")
        xml_files.append(xml_path)
    if gt_dir:
        if not gt_dir.exists():
            raise click.ClickException(f"Ground-truth directory not found: {gt_dir}")
        xml_files.extend(sorted(gt_dir.rglob("*.xml")))
    if not xml_files:
        raise click.ClickException("No XML files found. Provide --gt-dir or --gt-file.")
    return xml_files


def _collect_xml_files_from_manifest(
    manifest_ref: str,
    *,
    partition: str = "train",
) -> tuple[List[Path], str]:
    try:
        manifest = load_frozen_manifest(manifest_ref)
        cases = manifest.get_partition(partition)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    xml_files: List[Path] = []
    for case in cases:
        if case.xml_path is None:
            raise click.ClickException(
                f"Manifest case {case.id} does not define xml_path for partition {partition}."
            )
        if not case.xml_path.exists():
            raise click.ClickException(f"Ground-truth XML file not found: {case.xml_path}")
        xml_files.append(case.xml_path)

    if not xml_files:
        raise click.ClickException(
            f"Manifest {manifest.manifest_id} partition {partition} has no XML files."
        )
    return xml_files, manifest.manifest_id


def _print_json(payload: Dict[str, Any]) -> None:
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_output_formats(format_str: str) -> List[str]:
    return [f.strip().lower() for f in format_str.split(",") if f.strip()]


def _resolve_output_path(
    output_arg: str | None,
    input_path: Path,
    output_format: str,
) -> Path | None:
    if output_arg is None:
        return None

    output_path = Path(output_arg)
    if output_path.is_dir():
        suffix = ".md" if output_format == "markdown" else ".json"
        return output_path / f"{input_path.stem}{suffix}"
    return output_path


@click.group()
def main() -> None:
    """Sogdian manuscript HTR CLI using Kraken."""


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--lang", default="sogdian", show_default=True, type=LANG_CHOICES)
@click.option(
    "--model",
    "-m",
    help="Kraken .mlmodel path. Defaults to MSOCR_HTR_RUNTIME_MODEL_PATH when set.",
)
@click.option(
    "--variant",
    default="standard",
    show_default=True,
    help="Sogdian manuscript variant label recorded with the output.",
)
@click.option(
    "--output-format",
    "-f",
    type=OUTPUT_FORMAT_CHOICES,
    default="json",
    show_default=True,
    help="Output format: json or markdown.",
)
@click.option("--output", "-o", help="Output path (file or directory)")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def htr(input_path, lang, model, variant, output_format, output, device) -> None:
    """Run manuscript HTR for an image or PDF input."""
    from msocr.output.formats import _generate_markdown, save_output
    from msocr.service.runtime import run_htr_service
    from msocr.utils.input_loader import expand_input_to_images

    lang_key = _normalize_lang(lang)
    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    formats = _parse_output_formats(output_format)
    with tempfile.TemporaryDirectory(prefix="msocr-htr-") as td:
        try:
            image_paths = expand_input_to_images(input_path, Path(td))
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        page_results = []
        selected_engine = ENGINE_NAME
        for idx, image_path in enumerate(image_paths, start=1):
            try:
                response = run_htr_service(
                    lang=lang_key,
                    image_path=image_path,
                    model=model,
                    variant=variant,
                    device=device,
                )
            except Exception as exc:
                raise click.ClickException(str(exc)) from exc

            selected_engine = response["engine"]
            page_results.append(
                {
                    "page_number": idx,
                    "text": response["text"],
                    "image_path": str(image_path),
                }
            )

    click.echo(f"engine={selected_engine} mode=htr lang={lang_key} pages={len(page_results)}")

    output_data = {
        "mode": "htr",
        "writing_mode": "handwritten",
        "language": lang_key,
        "engine": selected_engine,
        "pages": page_results,
        "metadata": {
            "input_file": str(input_path),
            "variant": variant,
        },
    }

    for fmt in formats:
        output_path = _resolve_output_path(output, input_path, fmt)
        output_path = save_output(output_data, image_paths, fmt, output_path, language=lang_key)
        if output_path:
            click.echo(f"{fmt.upper()} result saved to {output_path}")
        elif fmt == "json":
            click.echo(json.dumps(output_data, ensure_ascii=False, indent=2))
        elif fmt == "markdown":
            click.echo(_generate_markdown(output_data))


@main.command()
@click.option("--lang", default="sogdian", show_default=True, type=LANG_CHOICES)
@click.option("--config", "-c", help="Training config file; defaults to Sogdian HTR config")
@click.option(
    "--gt-dir",
    type=click.Path(path_type=Path),
    help="Directory containing PAGE/ALTO XML files",
)
@click.option("--gt-file", help="Single PAGE/ALTO XML file for training")
@click.option(
    "--split-manifest-id",
    help="Frozen split manifest id or path used to resolve the training partition",
)
@click.option(
    "--split-partition",
    default="train",
    show_default=True,
    type=click.Choice(["train", "validation", "holdout"], case_sensitive=False),
    help="Manifest partition to use when --split-manifest-id is provided",
)
def train(lang, config, gt_dir, gt_file, split_manifest_id, split_partition) -> None:
    """Train a Kraken HTR model from manuscript PAGE/ALTO XML."""
    lang_key = _normalize_lang(lang)
    config_path = Path(config) if config else _resolve_default_config(lang_key)
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)

    manifest_label = None
    if split_manifest_id:
        if gt_dir or gt_file:
            raise click.ClickException(
                "Use either --split-manifest-id or --gt-dir/--gt-file, not both."
            )
        xml_files, manifest_label = _collect_xml_files_from_manifest(
            split_manifest_id,
            partition=split_partition,
        )
    else:
        xml_files = _collect_xml_files(gt_dir, gt_file)

    trainer = KetosTrainer(config_dict)
    ok = trainer.train(xml_files=xml_files)
    if not ok:
        raise click.ClickException("Training failed. See logs for details.")

    summary = (
        f"Training completed: engine={ENGINE_NAME} mode=htr lang={lang_key} "
        f"xml_files={len(xml_files)} config={config_path}"
    )
    if manifest_label:
        summary += f" split_manifest_id={manifest_label} partition={split_partition.lower()}"
    click.echo(summary)


@main.command()
@click.option("--input-dir", "-i", required=True, help="Input directory with manuscript images")
def preprocess(input_dir) -> None:
    """Preprocess manuscript images for Kraken HTR."""
    from msocr.preprocessing.preprocessor import preprocess_directory

    output_dir = Path(input_dir) / "processed"
    preprocess_directory(str(input_dir), str(output_dir))
    click.echo(f"Preprocessed images saved to {output_dir}")


def _parse_roi(roi: str | None) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    parts = [part.strip() for part in roi.split(",")]
    if len(parts) != 4:
        raise click.ClickException("--roi must be LEFT,TOP,RIGHT,BOTTOM")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise click.ClickException("--roi values must be integers") from exc


def _parse_row_centers(row_centers: str | None) -> list[int] | None:
    if row_centers is None:
        return None
    try:
        return [int(part.strip()) for part in row_centers.split(",") if part.strip()]
    except ValueError as exc:
        raise click.ClickException("--row-centers values must be integers") from exc


@main.command(name="extract-lines")
@click.argument("image", type=click.Path(path_type=Path, exists=True))
@click.option("--expected-lines", required=True, type=int, help="Exact manuscript line count to output")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path), help="Directory for crops and QA images")
@click.option("--roi", help="Optional text region as LEFT,TOP,RIGHT,BOTTOM")
@click.option("--row-centers", help="Optional comma-separated row center y-coordinates")
@click.option("--min-component-area", default=20, show_default=True, type=int, help="Ignore smaller ink flecks")
def extract_lines(image, expected_lines, output_dir, roi, row_centers, min_component_area) -> None:
    """Extract exact-count fragmented manuscript row crops."""
    from msocr.segmentation.row_bands import extract_row_bands

    bands = extract_row_bands(
        image,
        output_dir,
        expected_lines=expected_lines,
        roi=_parse_roi(roi),
        row_centers=_parse_row_centers(row_centers),
        min_component_area=min_component_area,
    )
    click.echo(f"Extracted {len(bands)} lines to {output_dir / 'lines'}")
    click.echo(f"Overlay: {output_dir / 'line_overlay.jpg'}")
    click.echo(f"Contact sheet: {output_dir / 'line_contact_sheet.jpg'}")


@main.command(name="api")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve_api(host, port, reload) -> None:
    """Run the FastAPI HTR service."""
    from msocr.service.deploy import run_api_server

    run_api_server(host=host, port=port, reload=reload)


@main.command(name="annotation-api")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default="8001", show_default=True, type=int, help="Bind port")
@click.option(
    "--base-dir",
    type=click.Path(path_type=Path),
    default=Path("msocr/data"),
    show_default=True,
    help="Base directory for persisted annotation sessions",
)
@click.option(
    "--no-crop-manuscript-area",
    is_flag=True,
    default=False,
    help="Disable auto-detection and cropping of manuscript area before line segmentation",
)
def serve_annotation_api(host, port, base_dir, no_crop_manuscript_area) -> None:
    """Run the dedicated annotation FastAPI service."""
    import uvicorn

    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=base_dir, crop_manuscript_area=not no_crop_manuscript_area)
    uvicorn.run(app, host=host, port=port)


@main.command(name="demo")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=7860, show_default=True, type=int, help="Bind port")
@click.option("--share", is_flag=True, help="Enable Gradio public sharing link")
def demo_gradio(host, port, share) -> None:
    """Run the Sogdian manuscript HTR demo."""
    from msocr.service.deploy import run_demo_server

    run_demo_server(host=host, port=port, share=share)


@main.command(name="runtime-smoke-check")
@click.option("--lang", default="sogdian", show_default=True, type=LANG_CHOICES)
@click.option(
    "--variant",
    default="standard",
    show_default=True,
    help="Sogdian manuscript variant label used for runtime selection",
)
@click.option(
    "--image",
    type=click.Path(path_type=Path, exists=True),
    help="Optional image path to run an end-to-end HTR smoke check",
)
@click.option("--model", type=click.Path(path_type=Path), help="Optional model override")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
@click.option(
    "--base-url",
    help="Optional live runtime base URL to probe over HTTP (for example http://127.0.0.1:8000)",
)
@click.option(
    "--timeout",
    default=120,
    show_default=True,
    type=int,
    help="Overall timeout in seconds when waiting for a live runtime health check or HTR probe",
)
@click.option(
    "--poll-interval",
    default=3,
    show_default=True,
    type=int,
    help="Polling interval in seconds when waiting for a live runtime health check",
)
@click.option("--require-engine", help="Optional engine name that the smoke response must report")
def runtime_smoke_check_command(
    lang,
    variant,
    image,
    model,
    device,
    base_url,
    timeout,
    poll_interval,
    require_engine,
) -> None:
    """Validate HTR runtime model resolution, optionally with a live route smoke run."""
    from msocr.service.deploy import runtime_htr_smoke_check, runtime_http_htr_smoke_check

    try:
        if base_url:
            payload = runtime_http_htr_smoke_check(
                base_url=base_url,
                language=_normalize_lang(lang),
                script_variant=variant,
                image_path=image,
                device=device,
                timeout_sec=timeout,
                poll_interval_sec=poll_interval,
            )
        else:
            payload = runtime_htr_smoke_check(
                language=_normalize_lang(lang),
                script_variant=variant,
                model=str(model) if model else None,
                image_path=image,
                device=device,
            )
        if require_engine:
            observed_engine = payload.get("engine") or payload.get("htr", {}).get("engine")
            if observed_engine != require_engine.lower():
                raise click.ClickException(
                    f"Expected runtime smoke engine {require_engine.lower()!r}, got {observed_engine!r}."
                )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_json(payload)


@main.command(name="train-remote")
@click.option("--manifest", required=True, type=click.Path(path_type=Path),
              help="Path to the frozen split manifest JSON")
@click.option("--style-group", required=True, help="style_group_id to train")
@click.option("--base-model", required=True, type=click.Path(path_type=Path),
              help="Path to base .safetensors model")
@click.option("--output-model", required=True, type=click.Path(path_type=Path),
              help="Path to write the fine-tuned .safetensors model")
@click.option("--reports-dir", default="reports/", show_default=True,
              type=click.Path(path_type=Path), help="Directory for evaluation reports")
@click.option("--pod-gpu", default="RTX 4090", show_default=True,
              help="RunPod GPU Cloud Pod GPU type id")
@click.option("--pod-image", default="msocr-kraken7:latest", show_default=True,
              help="RunPod pod Docker image")
@click.option("--ssh-key", default="~/.ssh/id_ed25519", show_default=True,
              help="SSH private key path for the pod")
@click.option("--epochs", default=50, show_default=True, type=int,
              help="Max ketos training epochs")
@click.option("--min-epochs", default=20, show_default=True, type=int,
              help="Minimum epochs before early stopping is allowed")
@click.option("--lag", default=10, show_default=True, type=int,
              help="Early-stop lag (epochs without val improvement)")
@click.option("--freeze-backbone", default=5000, show_default=True, type=int,
              help="Number of backbone params to freeze")
@click.option("--augment/--no-augment", default=True, show_default=True,
              help="Enable/disable ketos data augmentation")
@click.option("--device", default="cuda:0", show_default=True,
              help="Training device on the pod")
@click.option("--workers", default=8, show_default=True, type=int,
              help="Dataloader workers on the pod")
def train_remote(manifest, style_group, base_model, output_model, reports_dir,
                 pod_gpu, pod_image, ssh_key, epochs, min_epochs, lag,
                 freeze_backbone, augment, device, workers) -> None:
    """Train one style-group on a RunPod GPU Cloud Pod, then evaluate locally."""
    from msocr.training.orchestrator import walk_style_group
    from msocr.training.runpod_runner import RunPodRunner

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise click.ClickException(
            "RUNPOD_API_KEY environment variable is not set. "
            "Get one from https://console.runpod.io/tokens."
        )

    runner = RunPodRunner(
        api_key=api_key,
        image=pod_image,
        gpu_type=pod_gpu,
        ssh_key_path=os.path.expanduser(ssh_key),
    )
    try:
        report = walk_style_group(
            manifest_path=str(manifest),
            style_group_id=style_group,
            runner=runner,
            base_model_path=str(base_model),
            output_model_path=str(output_model),
            reports_dir=str(reports_dir),
            epochs=epochs,
            min_epochs=min_epochs,
            lag=lag,
            freeze_backbone=freeze_backbone,
            augment=augment,
            device=device,
            workers=workers,
        )
    except (RuntimeError, TimeoutError, FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Training complete. Fine-tuned model: {output_model}")
    click.echo(f"Report: {Path(report.get('report_path', reports_dir))}")


@main.command(name="evaluate")
@click.option("--manifest", required=True, type=click.Path(path_type=Path),
              help="Path to the frozen split manifest JSON")
@click.option("--style-group", required=True, help="style_group_id to evaluate")
@click.option("--model", required=True, type=click.Path(path_type=Path),
              help="Path to the .safetensors or .mlmodel model to evaluate")
@click.option("--reports-dir", default="reports/", show_default=True,
              type=click.Path(path_type=Path), help="Directory for evaluation reports")
def evaluate(manifest, style_group, model, reports_dir) -> None:
    """Run ketos test over a style-group's holdout partition and write a report."""
    from msocr.evaluation.harness import run_evaluation

    try:
        report = run_evaluation(
            manifest_path=str(manifest),
            style_group_id=style_group,
            model_path=str(model),
            reports_dir=str(reports_dir),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    report_files = sorted(Path(reports_dir).glob(f"{report['manifest_id']}__{style_group}__{Path(str(model)).stem}.*"))
    report_path = report_files[0] if report_files else Path(reports_dir)
    click.echo(f"Evaluation report: {report_path}")


@main.command(name="annotate")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default="8001", show_default=True, type=int, help="Bind port")
@click.option("--base-dir", default=".", show_default=True,
              type=click.Path(path_type=Path),
              help="Base directory for persisted annotation sessions")
@click.option(
    "--no-crop-manuscript-area",
    is_flag=True,
    default=False,
    help="Disable auto-detection and cropping of manuscript area before line segmentation",
)
def annotate(host, port, base_dir, no_crop_manuscript_area) -> None:
    """Run the annotation API and print the /ui URL.

    ponytail: prints the URL instead of auto-opening a browser — `webbrowser`
    is flaky in headless/Docker/SSH environments. The user can click the link.
    """
    import uvicorn

    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=base_dir, crop_manuscript_area=not no_crop_manuscript_area)
    sessions = sorted((base_dir / "sessions").glob("*/session.json"))
    suffix = f"/ui/{sessions[0].parent.name}" if sessions else "/ui/{session_id}"
    click.echo(f"Annotation UI: http://{host}:{port}{suffix}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
