"""Preprocessing module for manuscript OCR."""

from .pipeline import ManuscriptPreprocessor, preprocess_directory

__all__ = ['ManuscriptPreprocessor', 'preprocess_directory']