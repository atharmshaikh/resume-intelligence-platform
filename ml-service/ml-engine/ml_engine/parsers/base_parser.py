"""
Base interface for all document parsers.
Ensures consistent parser behavior across file types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
class BaseParser(ABC):
    """
    Base class for all document parsers.

    Provides:
    - File validation
    - Safe path handling
    - Consistent parser interface
    """
    def _validate_file(self, file_path: str | Path) -> Path:
        """
        Validate file path and return normalized Path object.
        """

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if not path.stat().st_size:
            raise ValueError(f"File is empty: {path}")

        return path   
     
    @abstractmethod
    def parse(self, file_path: str) -> str:
        """
        Extract raw text from a file.
        Must be implemented by subclasses.
        """
        raise NotImplementedError