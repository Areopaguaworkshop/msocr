"""Model inference for manuscript OCR."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Kraken imports - these will work when kraken is installed
try:
    from kraken import binarization
    from kraken.lib import models
    from kraken.rpred import rpred
    from kraken.blla import segment
    kraken_available = True
except ImportError:
    kraken_available = False
    logger.warning("Kraken not available. Install with: pip install kraken")


class OCRModel:
    """OCR model wrapper for trained Kraken models."""
    
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
        """Preprocess image for OCR."""
        try:
            # Use Kraken's built-in binarization
            from PIL import Image
            
            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Apply Kraken's binarization
                bin_img = binarization.nlbin(img)
                
                return bin_img
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            raise
    
    def predict_line(self, image_path: Path) -> Dict[str, Any]:
        """Predict text from a single line image."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Perform OCR
            result = rpred(self.model, processed_img, self.device)
            
            # Extract prediction
            predictions = []
            for line in result:
                predictions.append({
                    "text": line.prediction,
                    "confidence": getattr(line, 'confidence', 1.0),
                    "bounding_box": getattr(line, 'bbox', None)
                })
            
            return {
                "image_path": str(image_path),
                "predictions": predictions,
                "full_text": " ".join([p["text"] for p in predictions])
            }
            
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "predictions": [],
                "full_text": "",
                "error": str(e)
            }
    
    def predict_page(self, image_path: Path) -> Dict[str, Any]:
        """Predict text from a full page image with line segmentation."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Perform line segmentation using Kraken's blla
            lines = segment(processed_img)
            
            # Process each line
            all_predictions = []
            for line_img in lines:
                # Extract line region
                # Note: This is a simplified approach
                # In practice, you'd need proper line extraction
                
                # For now, return the segmented lines count
                all_predictions.append({
                    "line_number": len(all_predictions) + 1,
                    "status": "segmented"
                })
            
            return {
                "image_path": str(image_path),
                "lines_found": len(lines),
                "predictions": all_predictions,
                "note": "Full page processing requires additional implementation"
            }
            
        except Exception as e:
            logger.error(f"Error processing page {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "lines_found": 0,
                "predictions": [],
                "error": str(e)
            }


def predict(image_path: str, model_path: str, device: str = "cpu") -> str:
    """
    Convenience function for OCR prediction.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        device: Device to use for inference
    
    Returns:
        OCR result as text
    """
    try:
        # Load model
        model = OCRModel(model_path)
        model.set_device(device)
        
        # Perform prediction
        result = model.predict_line(Path(image_path))
        
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