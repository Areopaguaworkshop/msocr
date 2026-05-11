# %%
# Manuscript OCR Training Workflow for Old Turkish
# 
# This notebook provides a complete workflow for training Kraken models
# for Old Turkish manuscript OCR using the ketos training framework.

# %%
import marimo as mo
import subprocess
import yaml
from pathlib import Path
import sys
sys.path.append('/home/ajiap/project/msocr')

from msocr.data.manager import DatasetManager
from msocr.data.annotation import AnnotationExporter
from msocr.preprocessing.pipeline import preprocess_directory

# %%
# Configuration
CONFIG_PATH = "/home/ajiap/project/msocr/configs/old_turkish_config.yaml"
DATA_DIR = Path("/home/ajiap/project/msocr/data")
LOGS_DIR = Path("/home/ajiap/project/msocr/logs")

# %%
# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# %%
# Step 1: Dataset Setup
dataset_dir = mo.directory(path="dataset")
language = mo.text(value="old_turkish", placeholder="Enter language code")
manuscript_id = mo.text(value="", placeholder="Enter manuscript ID (optional)")

# %%
# Initialize dataset manager
dataset_manager = DatasetManager(DATA_DIR / "old_turkish_dataset")

# %%
# Step 2: Data Collection
print("=== Data Collection ===")
print("Please upload your Old Turkish manuscript images to the dataset directory.")
print(f"Expected location: {DATA_DIR / 'old_turkish_dataset' / 'images'}")

# %%
# Add images interactively
upload_dir = mo.directory(path="upload_images")

if upload_dir.exists():
    added_images = dataset_manager.add_directory(upload_dir, "old_turkish")
    print(f"Added {len(added_images)} images to the dataset")

# %%
# Display dataset statistics
stats = dataset_manager.get_statistics()
print(f"Dataset Statistics:")
print(f"Total images: {stats['total_images']}")
print(f"Languages: {stats['languages']}")
print(f"By status: {stats['by_status']}")

# %%
# Step 3: Preprocessing
print("=== Image Preprocessing ===")

if stats['total_images'] > 0:
    # Preprocess images
    raw_images_dir = dataset_manager.images_dir
    processed_dir = dataset_manager.processed_dir
    
    processed_count = preprocess_directory(str(raw_images_dir), str(processed_dir))
    print(f"Preprocessed {processed_count} images")
else:
    print("No images to preprocess. Please add images first.")

# %%
# Step 4: Annotation Setup
print("=== Annotation Setup ===")

# Initialize annotation exporter
annotation_exporter = AnnotationExporter(dataset_manager)

# Export for Label Studio
labelstudio_output = DATA_DIR / "labelstudio_export"
config_path, tasks_path = annotation_exporter.export_for_labelstudio(
    labelstudio_output, language="old_turkish"
)

print(f"Exported for Label Studio:")
print(f"Config: {config_path}")
print(f"Tasks: {tasks_path}")

# %%
# Step 5: Create training data compilation
print("=== Training Data Compilation ===")

# Get annotated images
image_paths, annotation_paths = dataset_manager.export_for_kraken(
    language="old_turkish", only_annotated=True
)

print(f"Found {len(image_paths)} annotated images")
print(f"Found {len(annotation_paths)} annotation files")

# %%
# Create binary dataset for Kraken
if len(annotation_paths) > 0:
    # Compile dataset
    dataset_arrow = DATA_DIR / "old_turkish_dataset.arrow"
    
    cmd = [
        "ketos", "compile", "-f", "xml", 
        "-o", str(dataset_arrow)
    ] + [str(path) for path in annotation_paths]
    
    print("Running ketos compile...")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Dataset compilation successful!")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling dataset: {e}")
        print(f"Error output: {e.stderr}")
else:
    print("No annotated images found. Please complete annotation first.")

# %%
# Step 6: Model Training
print("=== Model Training ===")

if dataset_arrow.exists():
    # Build ketos train command
    model_prefix = config['output']['model_prefix']
    model_spec = config['model']['spec']
    
    cmd = [
        "ketos", "train",
        "-f", "binary",
        "-s", model_spec,
        "-o", model_prefix,
        "--normalization", config['model']['normalization'],
        "--base-dir", config['model']['base_dir'],
        "--optimizer", config['training']['optimizer'],
        "--lrate", str(config['training']['learning_rate']),
        "--weight-decay", str(config['training']['weight_decay']),
        "--schedule", config['training']['schedule'],
        "--epochs", str(config['training']['epochs']),
        "--min-epochs", str(config['training']['min_epochs']),
        "--lag", str(config['training']['lag']),
        "--partition", str(config['training']['validation_split'])
    ]
    
    # Add augmentation if enabled
    if config['training'].get('augment', False):
        cmd.append("--augment")
    
    # Add device and precision settings
    cmd.extend([
        "--device", config['training']['device'],
        "--precision", config['training']['precision'],
        "--workers", str(config['training']['workers'])
    ])
    
    # Add dataset
    cmd.append(str(dataset_arrow))
    
    print("Training command:")
    print(" ".join(cmd))
    
    # Note: Uncomment the following lines to actually run training
    # print("Starting training...")
    # try:
    #     result = subprocess.run(cmd, check=True)
    #     print("Training completed successfully!")
    # except subprocess.CalledProcessError as e:
    #     print(f"Training failed: {e}")
else:
    print("No compiled dataset found. Please complete annotation and compilation first.")

# %%
# Step 7: Model Evaluation
print("=== Model Evaluation ===")

# Check if model exists
model_files = list(Path(".").glob(f"{model_prefix}*.mlmodel"))
if model_files:
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Found model: {latest_model}")
    
    # Test model
    test_cmd = [
        "ketos", "test",
        "-m", str(latest_model),
        "-f", "binary",
        str(dataset_arrow)
    ]
    
    print("Test command:")
    print(" ".join(test_cmd))
    
    # Uncomment to actually run testing
    # try:
    #     result = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
    #     print("Test results:")
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print(f"Testing failed: {e}")
else:
    print("No trained model found. Please complete training first.")

# %%
# Cloud Training Setup
print("=== Cloud Training Setup ===")

print("For cloud training, you can:")
print("1. Upload the compiled dataset.arrow file to cloud storage")
print("2. Use the same training commands on cloud GPU instances")
print("3. Download the trained model files back to local")

# Create cloud training script
cloud_script = DATA_DIR / "cloud_train.sh"
with open(cloud_script, 'w') as f:
    f.write(f"""#!/bin/bash
# Cloud training script for Old Turkish manuscript OCR

# Install dependencies
pip install kraken opencv-python scikit-image torch torchvision

# Download dataset (replace with your cloud storage URL)
# wget https://your-cloud-storage/old_turkish_dataset.arrow

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Run training
ketos train -f binary \\
    -s "{config['model']['spec']}" \\
    -o "{config['output']['model_prefix']}" \\
    --normalization "{config['model']['normalization']}" \\
    --base-dir "{config['model']['base_dir']}" \\
    --optimizer "{config['training']['optimizer']}" \\
    --lrate {config['training']['learning_rate']} \\
    --weight-decay {config['training']['weight_decay']} \\
    --schedule "{config['training']['schedule']}" \\
    --epochs {config['training']['epochs']} \\
    --min-epochs {config['training']['min_epochs']} \\
    --lag {config['training']['lag']} \\
    --partition {config['training']['validation_split']} \\
    {"--augment" if config['training'].get('augment', False) else ""} \\
    --device cuda \\
    --precision bf16-mixed \\
    --workers {config['training']['workers']} \\
    old_turkish_dataset.arrow

# Upload trained models (replace with your cloud storage URL)
# scp *.mlmodel user@your-cloud-storage:/models/
""")

print(f"Cloud training script created: {cloud_script}")

# %%
# Workflow Summary
print("=== Workflow Summary ===")
print("1. Dataset setup ✓")
print("2. Data collection ✓")
print("3. Preprocessing ✓")
print("4. Annotation setup ✓")
print("5. Data compilation ✓")
print("6. Model training (ready to run)")
print("7. Model evaluation (ready to run)")
print("8. Cloud training setup ✓")
print("\nNext steps:")
print("1. Upload manuscript images")
print("2. Complete annotation using Label Studio or another tool")
print("3. Import annotation results")
print("4. Run the training commands shown above")
print("5. Evaluate the trained model")
print("6. Optionally use cloud training for faster results")