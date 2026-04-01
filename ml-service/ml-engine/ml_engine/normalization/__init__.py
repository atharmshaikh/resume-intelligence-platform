"""
Normalization layer.

Responsible for transforming raw parsed resume sections
into a structured ATS-compatible schema.
"""

from .ats_builder import build_ats_structure

__all__ = [
    "build_ats_structure",
]