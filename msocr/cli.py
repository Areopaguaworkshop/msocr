"""CLI interface for msocr."""

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


@click.group()
def main():
    """Manuscript OCR CLI using the Kraken engine."""


@main.command()
@click.option("--lang", required=True, type=LANG_CHOICES, help="Target OCR language")
@click.option("--image", "-i", required=True, help="Input image path")
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
    "--variant",
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
@click.option("--output", "-o", help="Output text path")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def ocr(
    lang, image, model, engine, variant, reference_text, cer_threshold, output, device
):
    """Run printed OCR."""
    from msocr.pipelines.printed_ocr import run_printed_ocr

    image_path = Path(image)
    if not image_path.exists():
        raise click.ClickException(f"Input image not found: {image_path}")

    if model and not Path(model).exists():
        raise click.ClickException(f"Model file not found: {Path(model)}")

    pipeline_result = run_printed_ocr(
        lang=_normalize_lang(lang),
        image_path=image_path,
        model=model,
        device=device,
        engine=engine,
        variant=variant,
        reference_text_path=str(reference_text) if reference_text else None,
        cer_threshold=cer_threshold,
    )
    result = pipeline_result["text"]
    selected_engine = pipeline_result["engine"]
    click.echo(f"engine={selected_engine} mode=ocr lang={_normalize_lang(lang)}")
    if output:
        output_path = Path(output)
        output_path.write_text(result, encoding="utf-8")
        click.echo(f"OCR result saved to {output_path}")
    else:
        click.echo(result)


@main.command()
@click.option("--lang", required=True, type=LANG_CHOICES, help="Target HTR language")
@click.option("--image", "-i", required=True, help="Input image path")
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
@click.option("--output", "-o", help="Output text path")
@click.option("--device", default="cpu", show_default=True, help="Inference device")
def htr(lang, image, model, provider, output, device):
    """Run handwritten HTR."""
    from msocr.models.inference import predict

    lang_key = _normalize_lang(lang)
    provider_key = provider.lower()
    image_path = Path(image)
    if not image_path.exists():
        raise click.ClickException(f"Input image not found: {image_path}")

    if lang_key == "syriac" and provider_key in ("auto", "transkribus"):
        click.echo("provider=transkribus mode=htr lang=syriac")
        click.echo(
            "Syriac handwritten route is configured for Transkribus workflow currently. "
            "Export/import through Transkribus and then re-ingest results."
        )
        return

    if model:
        model_path = Path(model)
    elif lang_key == "latin":
        model_path = Path("models/kraken/latin_handwritten_mccatmus.mlmodel")
    elif lang_key == "greek":
        model_path = Path(
            "models/kraken/greek-german_serifs_sophokle1v3soph/"
            "greek-german_serifs_sophokle1v3soph.mlmodel"
        )
    else:
        raise click.ClickException(
            "HTR model is required for this language. Pass --model explicitly."
        )

    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")

    result = predict(str(image_path), str(model_path), device=device)
    click.echo(f"engine={ENGINE_NAME} mode=htr lang={lang_key}")
    if output:
        output_path = Path(output)
        output_path.write_text(result, encoding="utf-8")
        click.echo(f"HTR result saved to {output_path}")
    else:
        click.echo(result)


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


@main.command(name="benchmark-printed")
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


@main.command(name="serve-api")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve_api(host, port, reload):
    """Run FastAPI backend service."""
    import uvicorn

    uvicorn.run("msocr.service.api:app", host=host, port=port, reload=reload)


@main.command(name="demo-gradio")
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
