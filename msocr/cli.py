"""CLI interface for msocr."""

import tempfile
from pathlib import Path
from typing import Dict, List

import click
import yaml

from msocr.training.ketos_trainer import KetosTrainer

LANG_CHOICES = click.Choice(
    ["greek", "latin", "syriac", "coptic", "armenia", "geez", "sogdian", "old_turkish"],
    case_sensitive=False,
)
MODE_CHOICES = click.Choice(["ocr", "htr"], case_sensitive=False)
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
    type=click.Choice(["auto", "kraken", "tesseract"], case_sensitive=False),
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
            save_output(output_data, image_paths, fmt, output_path)
            click.echo(f"Searchable PDF saved to {output_path}")
        else:
            output_path = _resolve_output_path(output, input_path, fmt, "ocr")
            save_output(output_data, image_paths, fmt, output_path)
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
    "--output-format",
    "-f",
    type=OUTPUT_FORMAT_CHOICES,
    default="json",
    show_default=True,
    help="Output format: json or markdown (comma-separated for multiple, pdf not supported for HTR)",
)
@click.option("--output", "-o", help="Output path (file or directory)")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def htr(input_path, lang, model, provider, output_format, output, device):
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
        },
    }

    for fmt in formats:
        if fmt == "pdf":
            continue
        output_path = _resolve_output_path(output, input_path, fmt, "htr")
        save_output(output_data, image_paths, fmt, output_path)
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
def train(lang, mode, config, gt_dir, gt_file):
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

    xml_files = _collect_xml_files(gt_dir, gt_file)

    trainer = KetosTrainer(config_dict)
    ok = trainer.train(xml_files=xml_files)
    if not ok:
        raise click.ClickException("Training failed. See logs for details.")

    click.echo(
        f"Training completed: engine={ENGINE_NAME} mode={mode_key} "
        f"lang={lang_key} xml_files={len(xml_files)} config={config_path}"
    )


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
    required=True,
    type=click.Path(path_type=Path),
    help="Benchmark manifest path (.json or .jsonl)",
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
def benchmark_printed(manifest, output, cer_threshold):
    """Run printed OCR benchmark and write a JSON report."""
    from msocr.evaluation.printed_benchmark import run_printed_benchmark

    report = run_printed_benchmark(
        manifest_path=manifest,
        output_path=output,
        cer_threshold=cer_threshold,
    )
    click.echo(
        "benchmark=printed "
        f"total={report['total_cases']} ok={report['ok_cases']} "
        f"errors={report['error_cases']} pass_rate={report['pass_rate']:.3f}"
    )
    click.echo(f"report={output}")


@main.command(name="api")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve_api(host, port, reload):
    """Run FastAPI backend service."""
    import uvicorn

    uvicorn.run("msocr.service.api:app", host=host, port=port, reload=reload)


@main.command(name="demo")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=7860, show_default=True, type=int, help="Bind port")
@click.option(
    "--share", is_flag=True, help="Enable Gradio public sharing link (if supported)"
)
def demo_gradio(host, port, share):
    """Run Gradio browser demo."""
    from msocr.service.gradio_demo import build_demo

    demo = build_demo()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
