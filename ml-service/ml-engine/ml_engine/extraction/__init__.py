"""
Extraction layer.

Responsible for detecting resume sections and extracting
basic entities like name, email, phone, and location.
"""

from .entity_extractor import extract_entities
from .section_detector import detect_sections

__all__ = [
    "extract_entities",
    "detect_sections",
]