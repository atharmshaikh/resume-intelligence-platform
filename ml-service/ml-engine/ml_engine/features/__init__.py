"""
Feature extraction package for the Resume Intelligence Platform.

Responsible for converting structured resume data (ResumeSchema)
into numerical features used for scoring and ML models.
"""

from .feature_extractor import extract_features

__all__ = [
    "extract_features",
]