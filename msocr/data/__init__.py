"""Data management module for manuscript OCR."""

from .manifest import FrozenManifest, ManifestCase, load_frozen_manifest
from .manager import DatasetManager, create_dataset

__all__ = [
    "DatasetManager",
    "FrozenManifest",
    "ManifestCase",
    "create_dataset",
    "load_frozen_manifest",
]
