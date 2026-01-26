"""Data collection and management for manuscript OCR."""

import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages manuscript image datasets with metadata tracking."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.metadata_file = self.base_dir / "dataset_metadata.json"
        self.images_dir = self.base_dir / "images"
        self.annotations_dir = self.base_dir / "annotations"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "created": datetime.now().isoformat(),
                "images": {},
                "languages": {},
                "total_images": 0
            }
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def add_image(self, image_path: Path, language: str, 
                  manuscript_id: Optional[str] = None,
                  description: Optional[str] = None) -> str:
        """
        Add an image to the dataset.
        
        Returns the image ID assigned to the added image.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(image_path)
        
        # Check if image already exists
        for img_id, img_data in self.metadata["images"].items():
            if img_data.get("hash") == file_hash:
                logger.info(f"Image already exists with ID: {img_id}")
                return img_id
        
        # Generate unique image ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"{language}_{timestamp}_{image_path.stem}"
        
        # Copy image to dataset directory
        dest_path = self.images_dir / f"{image_id}{image_path.suffix}"
        shutil.copy2(image_path, dest_path)
        
        # Update metadata
        self.metadata["images"][image_id] = {
            "filename": dest_path.name,
            "original_path": str(image_path),
            "language": language,
            "manuscript_id": manuscript_id or "unknown",
            "description": description or "",
            "hash": file_hash,
            "added_date": datetime.now().isoformat(),
            "status": "uploaded",
            "annotated": False
        }
        
        # Update language counts
        if language not in self.metadata["languages"]:
            self.metadata["languages"][language] = 0
        self.metadata["languages"][language] += 1
        self.metadata["total_images"] += 1
        
        self._save_metadata()
        logger.info(f"Added image {image_id} to dataset")
        return image_id
    
    def add_directory(self, directory_path: Path, language: str,
                     manuscript_id: Optional[str] = None) -> List[str]:
        """Add all images from a directory to the dataset."""
        directory_path = Path(directory_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        added_ids = []
        for image_file in directory_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                try:
                    image_id = self.add_image(image_file, language, manuscript_id)
                    added_ids.append(image_id)
                except Exception as e:
                    logger.error(f"Failed to add {image_file}: {e}")
        
        logger.info(f"Added {len(added_ids)} images from {directory_path}")
        return added_ids
    
    def list_images(self, language: Optional[str] = None, 
                   status: Optional[str] = None) -> List[Dict]:
        """List images in the dataset with optional filtering."""
        images = []
        
        for image_id, image_data in self.metadata["images"].items():
            # Apply filters
            if language and image_data.get("language") != language:
                continue
            if status and image_data.get("status") != status:
                continue
            
            images.append({
                "id": image_id,
                **image_data
            })
        
        return images
    
    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the file path for an image ID."""
        if image_id not in self.metadata["images"]:
            return None
        
        filename = self.metadata["images"][image_id]["filename"]
        return self.images_dir / filename
    
    def update_image_status(self, image_id: str, status: str, 
                           annotated: bool = False):
        """Update the status of an image."""
        if image_id in self.metadata["images"]:
            self.metadata["images"][image_id]["status"] = status
            self.metadata["images"][image_id]["annotated"] = annotated
            self._save_metadata()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_images": self.metadata["total_images"],
            "languages": dict(self.metadata["languages"]),
            "by_status": {},
            "annotated_count": 0
        }
        
        for image_data in self.metadata["images"].values():
            status = image_data.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            if image_data.get("annotated", False):
                stats["annotated_count"] += 1
        
        return stats
    
    def export_for_kraken(self, language: Optional[str] = None, 
                         only_annotated: bool = False) -> Tuple[List[Path], List[Path]]:
        """
        Export image paths and annotation paths for Kraken training.
        
        Returns tuple of (image_paths, annotation_paths).
        """
        image_paths = []
        annotation_paths = []
        
        for image_id, image_data in self.metadata["images"].items():
            # Apply filters
            if language and image_data.get("language") != language:
                continue
            if only_annotated and not image_data.get("annotated", False):
                continue
            
            # Get paths
            image_path = self.get_image_path(image_id)
            annotation_path = self.annotations_dir / f"{image_id}.xml"
            
            if image_path and image_path.exists():
                image_paths.append(image_path)
                if annotation_path.exists():
                    annotation_paths.append(annotation_path)
        
        return image_paths, annotation_paths


def create_dataset(base_dir: str, language: str, 
                  manuscript_id: Optional[str] = None) -> DatasetManager:
    """Create a new dataset manager."""
    base_path = Path(base_dir) / f"{language}_dataset"
    if manuscript_id:
        base_path = base_path / manuscript_id
    
    return DatasetManager(base_path)