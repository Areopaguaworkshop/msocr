"""Image preprocessing helpers for manuscript enhancement."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ManuscriptPreprocessor:
    """Preprocessor for ancient manuscript images."""
    
    def __init__(self):
        self.target_dpi = 300
        self.min_contrast_threshold = 0.1
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to manuscript image."""
        # Non-local means denoising
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        # Additional median filter for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility."""
        # Convert to LAB color space for better contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary using adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use Otsu's method with adaptive approach
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew in manuscript image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            if angles:
                # Use median angle to correct skew
                median_angle = np.median(angles)
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                corrected = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return corrected
        
        return image
    
    def normalize_size(self, image: np.ndarray, target_size: Tuple[int, int] = (2000, 3000)) -> np.ndarray:
        """Normalize image size while maintaining aspect ratio."""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        if scale < 1.0:
            # Downsample if larger than target
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # Upsample if smaller than target
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def preprocess_single(self, image_path: Path, output_path: Path) -> bool:
        """Preprocess a single manuscript image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            logger.info(f"Processing {image_path.name}")
            
            # Apply preprocessing steps
            image = self.normalize_size(image)
            image = self.denoise(image)
            image = self.enhance_contrast(image)
            image = self.correct_skew(image)
            
            # Save both enhanced grayscale and binary versions
            enhanced_path = output_path / f"{image_path.stem}_enhanced.png"
            binary_path = output_path / f"{image_path.stem}_binary.png"
            
            # Save enhanced version
            cv2.imwrite(str(enhanced_path), image)
            
            # Save binary version
            binary = self.binarize(image)
            cv2.imwrite(str(binary_path), binary)
            
            logger.info(f"Saved processed images to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def preprocess_directory(self, input_dir: Path, output_dir: Path) -> int:
        """Preprocess all images in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        processed_count = 0
        
        for image_path in input_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                if self.preprocess_single(image_path, output_dir):
                    processed_count += 1
        
        logger.info(f"Processed {processed_count} images")
        return processed_count


def preprocess_directory(input_dir: str, output_dir: str) -> int:
    """Convenience function to preprocess a directory."""
    preprocessor = ManuscriptPreprocessor()
    return preprocessor.preprocess_directory(Path(input_dir), Path(output_dir))
