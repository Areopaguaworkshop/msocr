"""CLI interface for the Sogdian manuscript HTR pipeline."""

from __future__ import annotations

import json
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
    help="Output format: json or markdown. Comma-separated values are accepted.",
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
    from msocr.preprocessing.pipeline import preprocess_directory

    output_dir = Path(input_dir) / "processed"
    preprocess_directory(str(input_dir), str(output_dir))
    click.echo(f"Preprocessed images saved to {output_dir}")


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
@click.option("--port", default=8001, show_default=True, type=int, help="Bind port")
@click.option(
    "--base-dir",
    type=click.Path(path_type=Path),
    default=Path("msocr/data"),
    show_default=True,
    help="Base directory for persisted annotation sessions",
)
def serve_annotation_api(host, port, base_dir) -> None:
    """Run the dedicated annotation FastAPI service."""
    import uvicorn

    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=base_dir)
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


if __name__ == "__main__":
    main()
