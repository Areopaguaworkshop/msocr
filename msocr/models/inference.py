"""Model inference for Sogdian manuscript HTR."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Kraken imports - these will work when kraken is installed
try:
    from kraken import binarization
    from kraken.blla import segment
    from kraken.lib import models
    from kraken.rpred import rpred

    kraken_available = True
except ImportError:
    kraken_available = False
    logger.warning("Kraken not available. Install with: pip install kraken")


class OCRModel:
    """Kraken model wrapper for Sogdian manuscript HTR."""

    def __init__(self, model_path: Path):
        if not kraken_available:
            raise ImportError("Kraken is required but not installed")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self):
        """Load the Kraken model."""
        try:
            self.model = models.load_any(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def set_device(self, device: str):
        """Set inference device."""
        if device in ["cpu", "cuda", "cuda:0", "cuda:1"]:
            self.device = device
        else:
            raise ValueError(f"Unsupported device: {device}")

    def preprocess_image(self, image_path: Path) -> Any:
        """Preprocess image for Kraken recognition."""
        try:
            # Use Kraken's built-in binarization
            from PIL import Image

            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != "L":
                    img = img.convert("L")

                # Apply Kraken's binarization
                bin_img = binarization.nlbin(img)

                return bin_img
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            raise

    def predict_line(self, image_path: Path, segmentation_type: str = "baseline") -> Dict[str, Any]:
        """Predict text from a single line image.
        
        Args:
            image_path: Path to the image file
            segmentation_type: Type of segmentation to use - "baseline" for page/region images
                             or "bbox" for already cropped line images.
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
    
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
    
            from kraken.containers import Segmentation, BaselineLine, BBoxLine
    
            width, height = processed_img.size
            
            if segmentation_type == "baseline":
                # Use Kraken baseline segmentation before HTR recognition.
                # Step 1: Segment the page into lines using Kraken's blla
                logger.info("Performing baseline segmentation on page...")
                bounds = segment(
                    processed_img,
                    text_direction="horizontal-rl",
                    device=self.device,
                )

                # Step 2: Run HTR on segmented lines
                logger.info("Running HTR recognition on segmented lines...")
                result = rpred(self.model, processed_img, bounds)
            else:
                # Bounding box segmentation for already cropped Sogdian line images.
                # Add small padding to avoid "Line polygon outside of image bounds" error
                pad = 2
                bbox = (pad, pad, width - pad, height - pad)
                bounds = Segmentation(
                    type="bbox",
                    imagename=str(image_path),
                    text_direction="horizontal-rl",
                    script_detection=False,
                    lines=[BBoxLine(
                        id="0",
                        bbox=bbox
                    )],
                    regions=None,
                    line_orders=None,
                    language=None,
                )
                # Perform HTR with explicit segmentation bounds.
                result = rpred(self.model, processed_img, bounds)

            # Extract prediction
            predictions = []
            for line in result:
                predictions.append(
                    {
                        "text": line.prediction,
                        "confidence": getattr(line, "confidence", 1.0),
                        "bounding_box": getattr(line, "bbox", None),
                    }
                )

            return {
                "image_path": str(image_path),
                "predictions": predictions,
                "full_text": " ".join([p["text"] for p in predictions]),
            }

        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "predictions": [],
                "full_text": "",
                "error": str(e),
            }

    def predict_page(self, image_path: Path) -> Dict[str, Any]:
        """Predict text from a full page image with line segmentation."""
        return self.predict_line(image_path, segmentation_type="baseline")


def predict(image_path: str, model_path: str, device: str = "cpu", segmentation_type: str = "baseline") -> str:
    """
    Convenience function for HTR prediction.

    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        device: Device to use for inference
        segmentation_type: Type of segmentation - "baseline" or "bbox"

    Returns:
        HTR result as text
    """
    try:
        # Load model
        model = OCRModel(model_path)
        model.set_device(device)

        # Perform prediction
        result = model.predict_line(Path(image_path), segmentation_type=segmentation_type)

        # Return text
        return result.get("full_text", "")

    except Exception as e:
        logger.error(f"Error in predict function: {e}")
        return f"Error: {str(e)}"


def get_available_models(models_dir: Path) -> Dict[str, Path]:
    """Get list of available trained models."""
    models_dir = Path(models_dir)
    models = {}

    if not models_dir.exists():
        return models

    for model_file in models_dir.glob("*.mlmodel"):
        model_name = model_file.stem
        models[model_name] = model_file

    return models


def validate_model(model_path: Path) -> bool:
    """Validate if a model file is compatible."""
    try:
        if not kraken_available:
            return False

        model_path = Path(model_path)
        if not model_path.exists():
            return False

        # Try to load the model
        models.load_any(model_path)
        return True

    except Exception:
        return False
