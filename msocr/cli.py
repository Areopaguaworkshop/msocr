"""CLI interface for msocr."""

import click
from pathlib import Path


@click.group()
def main():
    """Manuscript OCR for Sogdian and Old Turkish."""
    pass


@main.command()
@click.option('--image', '-i', required=True, help='Input image path')
@click.option('--model', '-m', required=True, help='Model file path')
@click.option('--output', '-o', help='Output file path')
def ocr(image, model, output):
    """Run OCR on a manuscript image."""
    from msocr.models.inference import predict
    
    result = predict(image, model)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(result)
        click.echo(f"OCR result saved to {output}")
    else:
        click.echo(result)


@main.command()
@click.option('--input-dir', '-i', required=True, help='Input directory with images')
def preprocess(input_dir):
    """Preprocess manuscript images."""
    from msocr.preprocessing.pipeline import preprocess_directory
    
    output_dir = Path(input_dir) / "processed"
    preprocess_directory(str(input_dir), str(output_dir))
    click.echo(f"Preprocessed images saved to {output_dir}")


@main.command()
@click.option('--config', '-c', required=True, help='Training config file')
def train(config):
    """Train Kraken model."""
    import yaml
    from msocr.training.ketos_trainer import KetosTrainer
    
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = KetosTrainer(config)
    trainer.train()
    click.echo("Training completed")


if __name__ == '__main__':
    main()