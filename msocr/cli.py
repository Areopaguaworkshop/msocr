"""CLI interface for msocr."""

import json
import tempfile
from datetime import datetime, timezone
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

LANG_CHOICES = click.Choice(
    [*CLI_LANGUAGE_CODES, *CLI_LANGUAGE_ALIASES],
    case_sensitive=False,
)
MODE_CHOICES = click.Choice(["ocr", "htr"], case_sensitive=False)
RUNTIME_SMOKE_MODE_CHOICES = click.Choice(["printed", "handwritten"], case_sensitive=False)
OUTPUT_FORMAT_CHOICES = click.Choice(
    ["json", "pdf", "markdown"],
    case_sensitive=False,
)

ENGINE_NAME = "kraken"

DEFAULT_CONFIGS: Dict[str, Dict[str, str]] = {
    "sogdian": {
        "ocr": "configs/sogdian_config.yaml",
        "htr": "configs/sogdian_config.yaml",
    },
    "old_turkish": {
        "ocr": "configs/old_turkish_config.yaml",
        "htr": "configs/old_turkish_config.yaml",
    },
}


def _normalize_lang(lang: str) -> str:
    try:
        return normalize_language_code(lang)
    except KeyError:
        return lang.strip().lower()


def _resolve_default_config(lang: str, mode: str) -> Path:
    lang_key = _normalize_lang(lang)
    mode_key = mode.lower()
    lang_entry = DEFAULT_CONFIGS.get(lang_key, {})
    config_path = lang_entry.get(mode_key)
    if not config_path:
        raise click.ClickException(
            f"No default config for lang={lang_key}, mode={mode_key}. "
            "Pass --config explicitly."
        )
    resolved = Path(config_path)
    if not resolved.exists():
        raise click.ClickException(f"Default config does not exist: {resolved}")
    return resolved


def _collect_xml_files(gt_dir: Path, gt_file: str) -> List[Path]:
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


def _require_manifest_reference(
    manifest: Path | None,
    manifest_id: str | None,
) -> str | Path:
    if manifest and manifest_id:
        raise click.ClickException("Provide either --manifest or --manifest-id, not both.")
    if manifest is None and not manifest_id:
        raise click.ClickException("Provide one of --manifest or --manifest-id.")
    return manifest if manifest is not None else str(manifest_id)


def _default_pipeline_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d%H%M%S")


def _print_json(payload: Dict[str, Any]) -> None:
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_metadata_pairs(values: tuple[str, ...]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for raw_value in values:
        if "=" not in raw_value:
            raise click.ClickException(
                f"Invalid metadata entry {raw_value!r}. Expected KEY=VALUE."
            )
        key, value = raw_value.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException(
                f"Invalid metadata entry {raw_value!r}. Metadata key cannot be empty."
            )
        metadata[key] = value.strip()
    return metadata


def _parse_output_formats(format_str: str) -> List[str]:
    return [f.strip().lower() for f in format_str.split(",")]


def _resolve_output_path(
    output_arg: str | None,
    input_path: Path,
    output_format: str,
    mode: str,
) -> Path | None:
    if output_arg is None:
        return None

    output_path = Path(output_arg)
    if output_path.is_dir():
        base_name = input_path.stem
        ext = _get_extension_for_format(output_format, mode)
        return output_path / f"{base_name}{ext}"
    else:
        return output_path


def _get_extension_for_format(output_format: str, mode: str) -> str:
    if output_format == "pdf":
        return ".pdf"
    elif output_format == "markdown":
        return ".md"
    else:
        return ".json"


@click.group()
def main():
    """Manuscript OCR CLI using the Kraken engine."""


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--lang", required=True, type=LANG_CHOICES, help="Target OCR language")
@click.option(
    "--model",
    "-m",
    help="Model file path (optional; overrides default language model)",
)
@click.option(
    "--engine",
    type=click.Choice(["auto", "kraken", "tesseract", "ocrmypdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Printed OCR engine selection",
)
@click.option(
    "--syriac-variant",
    type=click.Choice(["default", "estrangela", "serto", "east"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Script variant hint (used for Syriac printed routing)",
)
@click.option(
    "--reference-text",
    type=click.Path(path_type=Path),
    help="Reference text file for CER-gated fallback decisions",
)
@click.option(
    "--cer-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="CER threshold for Syriac Serto/East trained-model trigger",
)
@click.option(
    "--output-format",
    "-f",
    type=OUTPUT_FORMAT_CHOICES,
    default="json",
    show_default=True,
    help="Output format: json, pdf, or markdown (comma-separated for multiple)",
)
@click.option("--output", "-o", help="Output path (file or directory)")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def ocr(
    input_path,
    lang,
    model,
    engine,
    syriac_variant,
    reference_text,
    cer_threshold,
    output_format,
    output,
    device,
):
    """Run printed OCR for image or PDF input."""
    from msocr.pipelines.printed_ocr import run_printed_ocr
    from msocr.utils.input_loader import expand_input_to_images
    from msocr.output.formats import save_output

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    if model and not Path(model).exists():
        raise click.ClickException(f"Model file not found: {Path(model)}")

    formats = _parse_output_formats(output_format)
    if "pdf" in formats and "pdf" not in _parse_output_formats("json,pdf,markdown"):
        formats = list(set(formats))

    with tempfile.TemporaryDirectory(prefix="msocr-ocr-") as td:
        try:
            image_paths = expand_input_to_images(input_path, Path(td))
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        page_results = []
        selected_engine = "unknown"
        for idx, image_path in enumerate(image_paths, start=1):
            pipeline_result = run_printed_ocr(
                lang=_normalize_lang(lang),
                image_path=image_path,
                model=model,
                device=device,
                engine=engine,
                variant=syriac_variant,
                reference_text_path=str(reference_text) if reference_text else None,
                cer_threshold=cer_threshold,
            )
            selected_engine = pipeline_result["engine"]
            page_results.append(
                {
                    "page_number": idx,
                    "text": pipeline_result["text"],
                    "image_path": str(image_path),
                }
            )

    click.echo(
        f"engine={selected_engine} mode=ocr lang={_normalize_lang(lang)} pages={len(page_results)}"
    )

    output_data = {
        "mode": "ocr",
        "language": _normalize_lang(lang),
        "engine": selected_engine,
        "pages": page_results,
        "metadata": {
            "input_file": str(input_path),
        },
    }

    for fmt in formats:
        if fmt == "pdf":
            output_path = _resolve_output_path(output, input_path, fmt, "ocr")
            original_pdf = input_path if input_path.suffix.lower() == ".pdf" else None
            output_path = save_output(
                output_data,
                image_paths,
                fmt,
                output_path,
                original_pdf_path=original_pdf,
                language=lang,
            )
            click.echo(f"Searchable PDF saved to {output_path}")
        else:
            output_path = _resolve_output_path(output, input_path, fmt, "ocr")
            output_path = save_output(
                output_data, image_paths, fmt, output_path, language=lang
            )
            if output_path:
                click.echo(f"{fmt.upper()} result saved to {output_path}")
            else:
                if fmt == "json":
                    import json

                    click.echo(json.dumps(output_data, ensure_ascii=False, indent=2))
                elif fmt == "markdown":
                    from msocr.output.formats import _generate_markdown

                    md_text = _generate_markdown(output_data)
                    click.echo(md_text)


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--lang", required=True, type=LANG_CHOICES, help="Target HTR language")
@click.option(
    "--model",
    "-m",
    help="Model file path (optional for Latin/Greek defaults)",
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "kraken", "transkribus"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Handwritten OCR provider selection",
)
@click.option(
    "--variant",
    default="default",
    show_default=True,
    help="Script variant used for handwritten runtime model selection",
)
@click.option(
    "--output-format",
    "-f",
    type=OUTPUT_FORMAT_CHOICES,
    default="json",
    show_default=True,
    help="Output format: json or markdown (comma-separated for multiple, pdf not supported for HTR)",
)
@click.option("--output", "-o", help="Output path (file or directory)")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def htr(input_path, lang, model, provider, variant, output_format, output, device):
    """Run handwritten HTR for image or PDF input."""
    from msocr.service.runtime import run_htr_service
    from msocr.utils.input_loader import expand_input_to_images
    from msocr.output.formats import save_output

    lang_key = _normalize_lang(lang)
    provider_key = provider.lower()
    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    formats = _parse_output_formats(output_format)
    if "pdf" in formats:
        click.echo("Warning: PDF format not supported for HTR, ignoring.")

    with tempfile.TemporaryDirectory(prefix="msocr-htr-") as td:
        try:
            image_paths = expand_input_to_images(input_path, Path(td))
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        page_results = []
        selected_engine = ENGINE_NAME
        for idx, image_path in enumerate(image_paths, start=1):
            try:
                resp = run_htr_service(
                    lang=lang_key,
                    image_path=image_path,
                    model=model,
                    provider=provider_key,
                    variant=variant,
                    device=device,
                )
            except Exception as exc:
                raise click.ClickException(str(exc)) from exc

            selected_engine = resp["engine"]
            page_results.append(
                {
                    "page_number": idx,
                    "text": resp["text"],
                    "image_path": str(image_path),
                }
            )

    click.echo(
        f"engine={selected_engine} mode=htr lang={lang_key} pages={len(page_results)}"
    )

    output_data = {
        "mode": "htr",
        "language": lang_key,
        "engine": selected_engine,
        "pages": page_results,
        "metadata": {
            "input_file": str(input_path),
            "variant": variant,
        },
    }

    for fmt in formats:
        if fmt == "pdf":
            continue
        output_path = _resolve_output_path(output, input_path, fmt, "htr")
        output_path = save_output(
            output_data, image_paths, fmt, output_path, language=lang_key
        )
        if output_path:
            click.echo(f"{fmt.upper()} result saved to {output_path}")
        else:
            if fmt == "json":
                import json

                click.echo(json.dumps(output_data, ensure_ascii=False, indent=2))
            elif fmt == "markdown":
                from msocr.output.formats import _generate_markdown

                md_text = _generate_markdown(output_data)
                click.echo(md_text)


@main.command()
@click.option(
    "--lang", required=True, type=LANG_CHOICES, help="Target training language"
)
@click.option(
    "--mode", required=True, type=MODE_CHOICES, help="Training mode: ocr or htr"
)
@click.option(
    "--config", "-c", help="Training config file; defaults by lang/mode when available"
)
@click.option(
    "--gt-dir", type=click.Path(path_type=Path), help="Directory containing XML files"
)
@click.option("--gt-file", help="Single XML file for training")
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
def train(lang, mode, config, gt_dir, gt_file, split_manifest_id, split_partition):
    """Train OCR/HTR models with Kraken ketos."""
    lang_key = _normalize_lang(lang)
    mode_key = mode.lower()

    config_path = (
        Path(config) if config else _resolve_default_config(lang_key, mode_key)
    )
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
        f"Training completed: engine={ENGINE_NAME} mode={mode_key} "
        f"lang={lang_key} xml_files={len(xml_files)} config={config_path}"
    )
    if manifest_label:
        summary += f" split_manifest_id={manifest_label} partition={split_partition.lower()}"
    click.echo(summary)


@main.command()
@click.option("--input-dir", "-i", required=True, help="Input directory with images")
def preprocess(input_dir):
    """Preprocess manuscript images."""
    from msocr.preprocessing.pipeline import preprocess_directory

    output_dir = Path(input_dir) / "processed"
    preprocess_directory(str(input_dir), str(output_dir))
    click.echo(f"Preprocessed images saved to {output_dir}")


@main.command(name="benchmark")
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    help="Benchmark manifest path (.json or .jsonl)",
)
@click.option(
    "--manifest-id",
    help="Frozen benchmark manifest id under data/manifests/ (or explicit path)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output/benchmarks/printed_report.json"),
    show_default=True,
    help="Benchmark report output path",
)
@click.option(
    "--cer-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="CER pass threshold for printed benchmark",
)
@click.option(
    "--benchmark-id",
    help="Stable benchmark identifier; defaults to the resolved manifest_id",
)
@click.option(
    "--model-id",
    default="printed_ocr",
    show_default=True,
    help="Model family or route identifier for metrics.json provenance",
)
@click.option(
    "--model-version",
    default="local",
    show_default=True,
    help="Model version recorded in the benchmark report",
)
@click.option(
    "--pipeline-run-id",
    default="local",
    show_default=True,
    help="Pipeline or orchestration run identifier recorded in the report",
)
@click.option(
    "--preprocessing-profile",
    default="default",
    show_default=True,
    help="Preprocessing profile label recorded in the report",
)
def benchmark_printed(
    manifest,
    manifest_id,
    output,
    cer_threshold,
    benchmark_id,
    model_id,
    model_version,
    pipeline_run_id,
    preprocessing_profile,
):
    """Run printed OCR benchmark and write a JSON report."""
    from msocr.evaluation.printed_benchmark import run_printed_benchmark

    manifest_ref = _require_manifest_reference(manifest, manifest_id)
    report = run_printed_benchmark(
        output_path=output,
        manifest_path=manifest_ref if isinstance(manifest_ref, Path) else None,
        manifest_id=manifest_ref if isinstance(manifest_ref, str) else None,
        cer_threshold=cer_threshold,
        benchmark_id=benchmark_id,
        model_id=model_id,
        model_version=model_version,
        preprocessing_profile=preprocessing_profile,
        pipeline_run_id=pipeline_run_id,
    )
    click.echo(
        "benchmark=printed "
        f"benchmark_id={report['benchmark_id']} manifest_id={report['manifest_id']} "
        f"total={report['total_cases']} ok={report['ok_cases']} "
        f"errors={report['error_cases']} pass_rate={report['pass_rate']:.3f} "
        f"pass={str(report['pass_fail']).lower()}"
    )
    click.echo(f"report={output}")


@main.command(name="runpod-submit")
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    help="Training split manifest path",
)
@click.option(
    "--manifest-id",
    help="Training split manifest id under data/manifests/ (or explicit path)",
)
@click.option("--lang", required=False, type=LANG_CHOICES, help="Training language")
@click.option(
    "--script-variant",
    default="default",
    show_default=True,
    help="Script variant label used for pod metadata and artifact naming",
)
@click.option(
    "--writing-mode",
    type=click.Choice(["printed", "handwritten"], case_sensitive=False),
    default="printed",
    show_default=True,
    help="Writing mode routed through the training job",
)
@click.option(
    "--mode",
    type=MODE_CHOICES,
    help="Training mode; defaults to ocr for printed and htr for handwritten",
)
@click.option(
    "--trainer",
    type=click.Choice(["auto", "kraken", "tesstrain", "tesseract"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Training backend hint used for RunPod GPU tier selection",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    help="Training config path mounted into the training image",
)
@click.option(
    "--split-partition",
    default="train",
    show_default=True,
    type=click.Choice(["train", "validation", "holdout"], case_sensitive=False),
    help="Manifest partition used to estimate corpus size and build the train command",
)
@click.option(
    "--pipeline-run-id",
    default="",
    help="Pipeline run identifier; defaults to a UTC timestamp-based value",
)
@click.option(
    "--model-version",
    default="local",
    show_default=True,
    help="Model version label passed into the training environment",
)
@click.option(
    "--training-image",
    default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    show_default=True,
    help="Training container image",
)
@click.option(
    "--gpu-tier",
    type=click.Choice(["auto", "rtx4090", "rtx3090", "a100"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="RunPod GPU tier selection",
)
@click.option(
    "--volume-gb",
    default=20,
    show_default=True,
    type=int,
    help="Persistent Pod volume size in GB",
)
@click.option(
    "--container-disk-gb",
    default=50,
    show_default=True,
    type=int,
    help="Container disk size in GB",
)
@click.option(
    "--interruptible",
    is_flag=True,
    help="Request a lower-cost interruptible pod instead of a reserved pod",
)
@click.option(
    "--network-volume-id",
    default="",
    help="Optional existing RunPod network volume id to attach",
)
@click.option(
    "--container-registry-auth-id",
    default="",
    help="Optional RunPod private registry auth id for training images",
)
@click.option(
    "--data-center-id",
    "data_center_ids",
    multiple=True,
    help="Preferred RunPod data center id. Repeat for priority order.",
)
@click.option(
    "--command",
    default="",
    help="Override the generated container command passed to bash -lc",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually submit the pod to RunPod. Without this flag the command prints the plan only.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="When used with --execute, poll until the pod reaches RUNNING",
)
def runpod_submit(
    manifest,
    manifest_id,
    lang,
    script_variant,
    writing_mode,
    mode,
    trainer,
    config,
    split_partition,
    pipeline_run_id,
    model_version,
    training_image,
    gpu_tier,
    volume_gb,
    container_disk_gb,
    interruptible,
    network_volume_id,
    container_registry_auth_id,
    data_center_ids,
    command,
    execute,
    wait,
):
    """Create a manifest-aware RunPod training job plan or submit it."""
    from msocr.pipeline.runpod_client import RunPodClient, build_training_job

    manifest_ref = _require_manifest_reference(manifest, manifest_id)
    try:
        resolved_manifest = load_frozen_manifest(manifest_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    language = _normalize_lang(lang or resolved_manifest.language or "")
    if not language:
        raise click.ClickException("Training language is required via --lang or manifest metadata.")

    writing_mode_key = writing_mode.lower()
    mode_key = mode.lower() if mode else ("ocr" if writing_mode_key == "printed" else "htr")
    trainer_key = trainer.lower()
    config_path = Path(config) if config else None
    run_id = pipeline_run_id.strip() or _default_pipeline_run_id()

    try:
        job = build_training_job(
            manifest=resolved_manifest,
            language=language,
            script_variant=script_variant,
            writing_mode=writing_mode_key,
            mode=mode_key,
            pipeline_run_id=run_id,
            model_version=model_version,
            training_image=training_image,
            training_backend=None if trainer_key == "auto" else trainer_key,
            config_path=config_path,
            partition=split_partition.lower(),
            gpu_tier=gpu_tier.lower(),
            command=command.strip() or None,
            volume_in_gb=volume_gb,
            container_disk_in_gb=container_disk_gb,
            interruptible=interruptible,
            network_volume_id=network_volume_id.strip() or None,
            container_registry_auth_id=container_registry_auth_id.strip() or None,
            data_center_ids=data_center_ids,
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    plan = {
        "manifest_id": resolved_manifest.manifest_id,
        "manifest_path": str(resolved_manifest.path),
        "pipeline_run_id": run_id,
        "language": language,
        "script_variant": script_variant,
        "writing_mode": writing_mode_key,
        "mode": mode_key,
        "corpus_size": len(resolved_manifest.get_partition(split_partition.lower())),
        "payload": job.to_api_payload(),
    }
    if not execute:
        _print_json(plan)
        return

    try:
        client = RunPodClient.from_env()
        pod = client.create_pod(job)
        response: Dict[str, Any] = {
            **plan,
            "submitted_pod": {
                "id": pod.pod_id,
                "name": pod.name,
                "desired_status": pod.desired_status,
                "public_ip": pod.public_ip,
                "port_mappings": pod.port_mappings,
            },
        }
        if wait:
            ready_pod = client.wait_for_status(pod.pod_id)
            response["submitted_pod"] = {
                "id": ready_pod.pod_id,
                "name": ready_pod.name,
                "desired_status": ready_pod.desired_status,
                "public_ip": ready_pod.public_ip,
                "port_mappings": ready_pod.port_mappings,
            }
        _print_json(response)
    except (RuntimeError, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command(name="har-publish")
@click.option("--registry", required=True, help="Harness Artifact Registry name")
@click.option("--lang", required=True, type=LANG_CHOICES, help="Model language")
@click.option(
    "--script-variant",
    default="default",
    show_default=True,
    help="Script variant used in artifact naming",
)
@click.option(
    "--writing-mode",
    type=click.Choice(["printed", "handwritten"], case_sensitive=False),
    default="printed",
    show_default=True,
    help="Writing mode used in artifact naming",
)
@click.option("--version", required=True, help="Artifact version or sequence id")
@click.option(
    "--model-file",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Primary model file to upload (.mlmodel or .traineddata)",
)
@click.option(
    "--metrics-file",
    type=click.Path(path_type=Path, exists=True),
    help="Optional metrics.json sidecar",
)
@click.option(
    "--config-file",
    type=click.Path(path_type=Path, exists=True),
    help="Optional training config sidecar",
)
@click.option(
    "--dockerfile-sha-file",
    type=click.Path(path_type=Path, exists=True),
    help="Optional Dockerfile.sha sidecar",
)
@click.option(
    "--pkg-url",
    default="https://pkg.harness.io",
    show_default=True,
    help="Harness Artifact Registry packages base URL",
)
@click.option("--description", default="", help="Optional artifact description")
@click.option(
    "--metadata",
    multiple=True,
    help="Artifact metadata entry in KEY=VALUE form. Repeat as needed.",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually invoke the Harness CLI. Without this flag the command prints the planned uploads only.",
)
def har_publish(
    registry,
    lang,
    script_variant,
    writing_mode,
    version,
    model_file,
    metrics_file,
    config_file,
    dockerfile_sha_file,
    pkg_url,
    description,
    metadata,
    execute,
):
    """Publish a model bundle and sidecars to Harness Artifact Registry."""
    from msocr.pipeline.har_client import HARClient, build_bundle

    metadata_map = _parse_metadata_pairs(tuple(metadata))
    bundle = build_bundle(
        registry=registry,
        language=_normalize_lang(lang),
        script_variant=script_variant,
        writing_mode=writing_mode.lower(),
        version=version,
        model_file=model_file,
        metrics_file=metrics_file,
        config_file=config_file,
        dockerfile_sha_file=dockerfile_sha_file,
        pkg_url=pkg_url,
        description=description.strip() or None,
        metadata=metadata_map,
    )
    client = HARClient(pkg_url=pkg_url)
    commands = [" ".join(command) for command in client.plan_commands(bundle)]
    plan = {
        "artifact_ref": bundle.artifact_ref,
        "registry": bundle.registry,
        "pkg_url": bundle.pkg_url,
        "files": [
            {
                "source_path": str(upload.source_path),
                "package_path": upload.package_path,
            }
            for upload in bundle.files
        ],
        "commands": commands,
    }
    if not execute:
        _print_json(plan)
        return

    try:
        client.publish_bundle(bundle)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_json({**plan, "published": True})


@main.command(name="pipeline-submit")
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    help="Training split manifest path",
)
@click.option(
    "--manifest-id",
    help="Training split manifest id under data/manifests/ (or explicit path)",
)
@click.option(
    "--benchmark-manifest",
    type=click.Path(path_type=Path),
    help="Benchmark manifest path used for evaluation and policy gating",
)
@click.option(
    "--benchmark-manifest-id",
    help="Benchmark manifest id under data/manifests/ (or explicit path)",
)
@click.option("--lang", required=False, type=LANG_CHOICES, help="Training language")
@click.option(
    "--script-variant",
    default="default",
    show_default=True,
    help="Script variant used for policy thresholds and artifact naming",
)
@click.option(
    "--writing-mode",
    type=click.Choice(["printed", "handwritten"], case_sensitive=False),
    default="printed",
    show_default=True,
    help="Writing mode routed through the training and promotion workflow",
)
@click.option(
    "--mode",
    type=MODE_CHOICES,
    help="Training mode; defaults to ocr for printed and htr for handwritten",
)
@click.option(
    "--trainer",
    type=click.Choice(["auto", "kraken", "tesstrain", "tesseract"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Training backend hint used for RunPod payloads and benchmark overrides",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    help="Training config path stored as a sidecar on promotion",
)
@click.option(
    "--pipeline-run-id",
    default="",
    help="Pipeline run identifier; defaults to a UTC timestamp-based value",
)
@click.option(
    "--sequence-id",
    default="",
    help="Artifact sequence identifier used to build HAR version v<sequence-id>",
)
@click.option(
    "--model-version",
    default="local",
    show_default=True,
    help="Model version recorded in metrics and training environment",
)
@click.option(
    "--model-file",
    type=click.Path(path_type=Path, exists=True),
    help="Local trained model artifact to benchmark and publish (.mlmodel or .traineddata)",
)
@click.option(
    "--runpod-model-path",
    default="",
    help="Remote model artifact path on the RunPod /workspace volume to retrieve when --model-file is not provided",
)
@click.option(
    "--runpod-ssh-key",
    type=click.Path(path_type=Path, exists=True),
    envvar="RUNPOD_SSH_KEY_PATH",
    help="Private SSH key used for rsync-based RunPod model retrieval",
)
@click.option(
    "--runpod-ssh-public-key",
    envvar="RUNPOD_SSH_PUBLIC_KEY",
    default="",
    help="Optional SSH public key injected into the pod as SSH_PUBLIC_KEY for automated retrieval",
)
@click.option(
    "--runpod-retrieve-timeout-sec",
    default=1800,
    show_default=True,
    type=int,
    help="Maximum seconds to wait for the remote RunPod model artifact to become retrievable",
)
@click.option(
    "--runpod-retrieve-poll-sec",
    default=15,
    show_default=True,
    type=int,
    help="Polling interval in seconds while retrying RunPod model retrieval",
)
@click.option(
    "--registry",
    default="",
    help="Harness Artifact Registry name used for promotion",
)
@click.option(
    "--pkg-url",
    default="https://pkg.harness.io",
    show_default=True,
    help="Harness Artifact Registry packages base URL",
)
@click.option(
    "--training-image",
    default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    show_default=True,
    help="Training container image",
)
@click.option(
    "--gpu-tier",
    type=click.Choice(["auto", "rtx4090", "rtx3090", "a100"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="RunPod GPU tier selection",
)
@click.option(
    "--volume-gb",
    default=20,
    show_default=True,
    type=int,
    help="Persistent Pod volume size in GB",
)
@click.option(
    "--container-disk-gb",
    default=50,
    show_default=True,
    type=int,
    help="Container disk size in GB",
)
@click.option(
    "--interruptible",
    is_flag=True,
    help="Request a lower-cost interruptible pod instead of a reserved pod",
)
@click.option(
    "--network-volume-id",
    default="",
    help="Optional existing RunPod network volume id to attach",
)
@click.option(
    "--container-registry-auth-id",
    default="",
    help="Optional RunPod private registry auth id for training images",
)
@click.option(
    "--data-center-id",
    "data_center_ids",
    multiple=True,
    help="Preferred RunPod data center id. Repeat for priority order.",
)
@click.option(
    "--dockerfile",
    type=click.Path(path_type=Path),
    default=Path("Dockerfile"),
    show_default=True,
    help="Dockerfile used to build the training image; its SHA is attached on promotion",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output/pipelines"),
    show_default=True,
    help="Output directory for pipeline metrics and generated sidecars",
)
@click.option(
    "--benchmark-id",
    default="",
    help="Stable benchmark identifier; defaults to the benchmark manifest id",
)
@click.option(
    "--model-id",
    default="",
    help="Model family identifier recorded in metrics.json",
)
@click.option(
    "--preprocessing-profile",
    default="default",
    show_default=True,
    help="Preprocessing profile recorded in metrics.json",
)
@click.option(
    "--cer-threshold",
    type=float,
    help="Optional override for policy gate CER threshold",
)
@click.option(
    "--description",
    default="",
    help="Optional artifact description for HAR promotion",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Artifact metadata entry in KEY=VALUE form. Repeat as needed.",
)
@click.option(
    "--notification-url",
    envvar="MSOCR_NOTIFICATION_URL",
    default="",
    help="Optional webhook URL that receives the final pipeline notification payload",
)
@click.option(
    "--command",
    default="",
    help="Override the generated RunPod container command passed to bash -lc",
)
@click.option(
    "--wait",
    is_flag=True,
    help="When used with --execute, poll the submitted RunPod pod until it reaches RUNNING; retrieval enables this automatically",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Execute the full workflow. Without this flag the command prints the end-to-end plan only.",
)
def pipeline_submit(
    manifest,
    manifest_id,
    benchmark_manifest,
    benchmark_manifest_id,
    lang,
    script_variant,
    writing_mode,
    mode,
    trainer,
    config,
    pipeline_run_id,
    sequence_id,
    model_version,
    model_file,
    runpod_model_path,
    runpod_ssh_key,
    runpod_ssh_public_key,
    runpod_retrieve_timeout_sec,
    runpod_retrieve_poll_sec,
    registry,
    pkg_url,
    training_image,
    gpu_tier,
    volume_gb,
    container_disk_gb,
    interruptible,
    network_volume_id,
    container_registry_auth_id,
    data_center_ids,
    dockerfile,
    output_dir,
    benchmark_id,
    model_id,
    preprocessing_profile,
    cer_threshold,
    description,
    metadata,
    notification_url,
    command,
    wait,
    execute,
):
    """Plan or execute a manifest-aware training, benchmark, and promotion workflow."""
    from msocr.pipeline.workflow import run_training_promotion_workflow

    train_manifest_ref = _require_manifest_reference(manifest, manifest_id)
    benchmark_manifest_ref: str | Path | None = None
    if benchmark_manifest or benchmark_manifest_id:
        benchmark_manifest_ref = _require_manifest_reference(
            benchmark_manifest,
            benchmark_manifest_id,
        )

    try:
        resolved_train_manifest = load_frozen_manifest(train_manifest_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    language = _normalize_lang(lang or resolved_train_manifest.language or "")
    if not language:
        raise click.ClickException("Training language is required via --lang or manifest metadata.")

    writing_mode_key = writing_mode.lower()
    mode_key = mode.lower() if mode else ("ocr" if writing_mode_key == "printed" else "htr")
    run_id = pipeline_run_id.strip() or _default_pipeline_run_id()
    metadata_map = _parse_metadata_pairs(tuple(metadata))

    try:
        result = run_training_promotion_workflow(
            train_manifest_ref=train_manifest_ref,
            benchmark_manifest_ref=benchmark_manifest_ref,
            language=language,
            script_variant=script_variant,
            writing_mode=writing_mode_key,
            mode=mode_key,
            trainer=trainer,
            config_path=Path(config) if config else None,
            pipeline_run_id=run_id,
            sequence_id=sequence_id.strip() or None,
            model_version=model_version,
            model_file=model_file,
            runpod_model_path=runpod_model_path.strip() or None,
            runpod_ssh_key_path=runpod_ssh_key,
            runpod_ssh_public_key=runpod_ssh_public_key.strip() or None,
            runpod_retrieve_timeout_sec=runpod_retrieve_timeout_sec,
            runpod_retrieve_poll_interval_sec=runpod_retrieve_poll_sec,
            registry=registry.strip() or None,
            training_image=training_image,
            gpu_tier=gpu_tier.lower(),
            volume_in_gb=volume_gb,
            container_disk_in_gb=container_disk_gb,
            interruptible=interruptible,
            network_volume_id=network_volume_id.strip() or None,
            container_registry_auth_id=container_registry_auth_id.strip() or None,
            data_center_ids=data_center_ids,
            dockerfile_path=dockerfile if dockerfile and dockerfile.exists() else None,
            output_dir=output_dir,
            cer_threshold=cer_threshold,
            benchmark_id=benchmark_id.strip() or None,
            model_id=model_id.strip() or None,
            preprocessing_profile=preprocessing_profile,
            pkg_url=pkg_url,
            description=description.strip() or None,
            metadata=metadata_map,
            notification_url=notification_url.strip() or None,
            command=command.strip() or None,
            wait_for_runpod=wait,
            execute=execute,
        )
    except (RuntimeError, TimeoutError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    _print_json(result)


@main.command(name="api")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve_api(host, port, reload):
    """Run FastAPI backend service."""
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
def serve_annotation_api(host, port, base_dir):
    """Run the dedicated annotation FastAPI service."""
    import uvicorn

    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=base_dir)
    uvicorn.run(app, host=host, port=port)


@main.command(name="demo")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=7860, show_default=True, type=int, help="Bind port")
@click.option(
    "--share", is_flag=True, help="Enable Gradio public sharing link (if supported)"
)
def demo_gradio(host, port, share):
    """Run Gradio browser demo."""
    from msocr.service.deploy import run_demo_server

    run_demo_server(host=host, port=port, share=share)


@main.command(name="runtime-smoke-check")
@click.option(
    "--mode",
    default="printed",
    show_default=True,
    type=RUNTIME_SMOKE_MODE_CHOICES,
    help="Runtime route to validate",
)
@click.option("--lang", required=True, type=LANG_CHOICES, help="Runtime language")
@click.option(
    "--variant",
    default="default",
    show_default=True,
    help="Script variant used for runtime route selection",
)
@click.option(
    "--image",
    type=click.Path(path_type=Path, exists=True),
    help=(
        "Optional image path to run an end-to-end printed OCR smoke check. "
        "When --base-url is set and --image is omitted, the canonical runtime smoke image is used."
    ),
)
@click.option(
    "--engine",
    type=click.Choice(["auto", "kraken", "tesseract", "ocrmypdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Printed OCR engine to use when --image is provided",
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "kraken", "transkribus"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Handwritten provider to use when --mode handwritten",
)
@click.option("--device", default="cpu", show_default=True, help="Inference device")
@click.option(
    "--reference-text",
    type=click.Path(path_type=Path, exists=True),
    help="Optional reference text used for CER-gated printed routing during smoke tests",
)
@click.option(
    "--cer-threshold",
    default=0.05,
    show_default=True,
    type=float,
    help="CER threshold used during the printed OCR smoke check",
)
@click.option(
    "--base-url",
    help="Optional live runtime base URL to probe over HTTP (for example http://127.0.0.1:8000)",
)
@click.option(
    "--timeout",
    default=120,
    show_default=True,
    type=int,
    help="Overall timeout in seconds when waiting for a live runtime health check or OCR probe",
)
@click.option(
    "--poll-interval",
    default=3,
    show_default=True,
    type=int,
    help="Polling interval in seconds when waiting for a live runtime health check",
)
def runtime_smoke_check_command(
    mode,
    lang,
    variant,
    image,
    engine,
    provider,
    device,
    reference_text,
    cer_threshold,
    base_url,
    timeout,
    poll_interval,
):
    """Validate deploy-time runtime model resolution, optionally with a live route smoke run."""
    from msocr.service.deploy import (
        runtime_htr_smoke_check,
        runtime_http_htr_smoke_check,
        runtime_http_smoke_check,
        runtime_smoke_check,
    )

    try:
        mode_key = mode.lower()
        if mode_key == "handwritten":
            if base_url:
                payload = runtime_http_htr_smoke_check(
                    base_url=base_url,
                    language=_normalize_lang(lang),
                    script_variant=variant,
                    image_path=image,
                    provider=provider.lower(),
                    device=device,
                    timeout_sec=timeout,
                    poll_interval_sec=poll_interval,
                )
            else:
                payload = runtime_htr_smoke_check(
                    language=_normalize_lang(lang),
                    script_variant=variant,
                    image_path=image,
                    provider=provider.lower(),
                    device=device,
                )
        elif base_url:
            payload = runtime_http_smoke_check(
                base_url=base_url,
                language=_normalize_lang(lang),
                script_variant=variant.lower(),
                image_path=image,
                engine=engine.lower(),
                device=device,
                reference_text_path=str(reference_text) if reference_text else None,
                cer_threshold=cer_threshold,
                timeout_sec=timeout,
                poll_interval_sec=poll_interval,
            )
        else:
            payload = runtime_smoke_check(
                language=_normalize_lang(lang),
                script_variant=variant.lower(),
                image_path=image,
                engine=engine.lower(),
                device=device,
                reference_text_path=str(reference_text) if reference_text else None,
                cer_threshold=cer_threshold,
            )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_json(payload)


@main.command(name="payne-smith")
@click.option(
    "--config",
    default="pipeline/payne-smith_syriac_runpod_train.yaml",
    show_default=True,
    help="Path to Payne-Smith pipeline YAML",
)
@click.option(
    "--runpod-config",
    default="",
    show_default=True,
    help="Path to unified RunPod training YAML",
)
@click.option(
    "--phases",
    default="phase_1a_validate_vienna_gt,phase_1b_split_vienna_dataset,phase_1c_pretrain,phase_1d_evaluate_vienna,phase_2a_ingest_pages,phase_2b_synthetic_augmentation,phase_2c_preprocess,phase_2d_segmentation,phase_2e_line_extraction,phase_2f_bootstrap_ocr,phase_2g_manual_correction,phase_2h_dataset_split,phase_2i_finetune_serto,phase_2j_finetune_estrangela,phase_2l_language_correction,stage_3_evaluation",
    show_default=True,
    help="Comma-separated phase list (e.g. 1,2,3,4)",
)
@click.option(
    "--input-pdf",
    type=click.Path(path_type=Path),
    help="Input PDF for phase 1 ingestion",
)
@click.option(
    "--workdir",
    type=click.Path(path_type=Path),
    default=".",
    show_default=True,
    help="Working directory for relative paths",
)
@click.option("--execute", is_flag=True, help="Execute actions (default is dry-run)")
def payne_smith_pipeline(config, runpod_config, phases, input_pdf, workdir, execute):
    """Run Payne-Smith Syriac OCR training pipeline."""
    from msocr.pipelines.payne_smith import PayneSmithPipeline

    phase_list = [p.strip() for p in phases.split(",") if p.strip()]
    ingest_aliases = {"1", "phase_2a_ingest_pages"}
    if any(p in ingest_aliases for p in phase_list) and not input_pdf:
        click.echo("Note: no --input-pdf provided, phase_2a will use input/payne_smith.pdf if present.")

    pipeline = PayneSmithPipeline(
        config_path=Path(config),
        runpod_path=Path(runpod_config) if runpod_config else None,
        workdir=Path(workdir),
        execute=execute,
    )
    results = pipeline.run(phase_list, input_pdf=input_pdf)
    for res in results:
        status = "ok" if res.ok else "error"
        click.echo(f"{res.phase}: {status} | {res.details}")


@main.command(name="runpod-train")
def runpod_train_reference():
    """Print RunPod training reference (markdown)."""
    ref_path = Path("pipeline/runpod_train_reference.md")
    if not ref_path.exists():
        raise click.ClickException(f"Reference not found: {ref_path}")
    click.echo(ref_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
