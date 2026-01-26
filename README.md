# Manuscript OCR for Sogdian and Old Turkish using Kraken

A comprehensive OCR system for ancient manuscript recognition, specifically designed for Sogdian and Old Turkish scripts using Kraken ketos training framework.

## Features

- **Multi-language Support**: Optimized for Sogdian and Old Turkish manuscripts
- **Advanced Preprocessing**: Noise reduction, contrast enhancement, skew correction
- **Standard Data Formats**: Full support for ALTO and PAGE XML standards
- **Training Workflow**: Complete training pipeline with marimo notebooks
- **Annotation Integration**: Export to Label Studio and CVAT for ground truth creation
- **Command-line Interface**: Easy-to-use CLI for OCR operations
- **Cloud Training Ready**: Support for GPU training on cloud platforms

## Installation

```bash
# Clone repository
git clone <repository-url>
cd msocr

# Install dependencies using uv
uv install

# Optional: Install development dependencies
uv install --dev
```

## Quick Start

### 1. Dataset Setup

```bash
# Create a dataset for Sogdian manuscripts
mkdir -p data/sogdian_dataset/images
# Copy your manuscript images to data/sogdian_dataset/images/

# Initialize dataset
python -c "
from msocr.data import DatasetManager
ds = DatasetManager('data/sogdian_dataset')
ds.add_directory('path/to/your/sogdian_images', 'sogdian')
"
```

### 2. Preprocess Images

```bash
# Preprocess all images in a directory
msocr preprocess --input-dir data/sogdian_dataset/images
```

### 3. Annotation Setup

```bash
# Export for Label Studio annotation
python -c "
from msocr.data import DatasetManager, AnnotationExporter
ds = DatasetManager('data/sogdian_dataset')
exporter = AnnotationExporter(ds)
exporter.export_for_labelstudio('labelstudio_export', language='sogdian')
"
```

### 4. Training

```bash
# Use marimo notebook for training
marimo edit notebooks/sogdian_training.py

# Or use CLI
msocr train --config configs/sogdian_config.yaml
```

### 5. OCR Inference

```bash
# Run OCR on a manuscript image
msocr ocr --image path/to/manuscript.jpg --model models/sogdian_manuscript.mlmodel
```

## Data Formats

### ALTO XML Format
The system supports ALTO XML version 4.2+ with the following structure:

```xml
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description>
    <sourceImageInformation>
      <fileName>manuscript.jpg</fileName>
    </sourceImageInformation>
  </Description>
  <Layout>
    <Page>
      <PrintSpace>
        <TextBlock ID="block_1">
          <TextLine ID="line_1" HPOS="100" VPOS="200" WIDTH="800" HEIGHT="50"
                    BASELINE="100 220 900 220">
            <String CONTENT="transcribed text"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
```

### PAGE XML Format
Also supports PAGE XML 2019-07-15 format:

```xml
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="manuscript.jpg">
    <TextRegion id="region_1">
      <Coords points="100,200 900,200 900,250 100,250"/>
      <TextLine id="line_1">
        <Baseline points="100,220 900,220"/>
        <TextEquiv><Unicode>transcribed text</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

## Training Configuration

### Sogdian Configuration
- **Model Spec**: Optimized for complex script with multiple diacritics
- **Direction**: Left-to-right
- **Normalization**: NFC Unicode normalization
- **Augmentation**: Enabled with rotation, scaling, and elastic distortion

### Old Turkish Configuration
- **Model Spec**: Optimized for Arabic-derived cursive script
- **Direction**: Right-to-left
- **Normalization**: NFC Unicode normalization
- **Augmentation**: Enhanced for cursive scripts with perspective distortion

## Cloud Training

For GPU training on cloud platforms:

```bash
# Upload compiled dataset
gsutil cp dataset.arrow gs://your-bucket/

# Use cloud training script
bash data/cloud_train.sh

# Download trained models
gsutil cp gs://your-bucket/*.mlmodel models/
```

## Project Structure

```
msocr/
├── msocr/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── data/                  # Data management
│   │   ├── __init__.py
│   │   ├── manager.py         # Dataset manager
│   │   └── annotation.py      # Annotation export/import
│   ├── preprocessing/          # Image preprocessing
│   │   ├── __init__.py
│   │   └── pipeline.py        # Preprocessing pipeline
│   ├── training/              # Model training
│   │   ├── __init__.py
│   │   └── ketos_trainer.py  # Kraken training wrapper
│   └── models/               # Model inference
│       ├── __init__.py
│       └── inference.py       # OCR inference
├── configs/
│   ├── sogdian_config.yaml    # Sogdian training config
│   └── old_turkish_config.yaml # Old Turkish training config
├── notebooks/
│   ├── sogdian_training.py    # Sogdian training notebook
│   └── old_turkish_training.py # Old Turkish training notebook
├── data/                     # Data directory
├── models/                   # Trained models
├── logs/                     # Training logs
└── pyproject.toml           # Project configuration
```

## Usage Examples

### Data Collection

```python
from msocr.data import DatasetManager

# Create dataset manager
ds = DatasetManager("data/my_dataset")

# Add single image
image_id = ds.add_image("path/to/manuscript.jpg", "sogdian", 
                        manuscript_id="manuscript_001",
                        description="Page 1 of manuscript")

# Add directory of images
added_ids = ds.add_directory("path/to/images/", "sogdian")

# Get dataset statistics
stats = ds.get_statistics()
print(f"Total images: {stats['total_images']}")
```

### Preprocessing

```python
from msocr.preprocessing import preprocess_directory

# Preprocess all images
processed_count = preprocess_directory("raw_images/", "processed_images/")
```

### Training

```python
import yaml
from msocr.training import KetosTrainer

# Load configuration
with open("configs/sogdian_config.yaml") as f:
    config = yaml.safe_load(f)

# Create trainer
trainer = KetosTrainer(config)

# Train model
trainer.train(xml_files)
```

### OCR Inference

```python
from msocr.models import OCRModel, predict

# Load model
model = OCRModel("models/sogdian_manuscript.mlmodel")

# Predict text from image
result = model.predict_line("path/to/manuscript_line.jpg")
print(f"Text: {result['full_text']}")

# Or use convenience function
text = predict("image.jpg", "model.mlmodel")
```

## Requirements

- Python 3.8+
- Kraken 5.0.0+
- OpenCV 4.8.0+
- PIL 10.0.0+
- NumPy 1.24.0+
- scikit-image 0.21.0+
- PyTorch 2.0.0+
- marimo 0.1.0+ (for notebooks)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Kraken OCR framework for the underlying recognition engine
- The manuscript community for feedback and testing
- Digital humanities researchers for requirements and validation