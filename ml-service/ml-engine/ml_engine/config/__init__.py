"""
Configuration layer facade.
Exposes engine constants seamlessly.
"""

from .settings import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from .ats_config import SECTION_KEYWORDS

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "MAX_FILE_SIZE_MB",
    "SECTION_KEYWORDS",
]
