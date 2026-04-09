"""
Base interface for all document parsers.

Provides:
- File validation
- Safe path handling
- Consistent parser interface
- Logging and error handling
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Base class for all document parsers.
    
    Provides common functionality for file validation and parsing.
    """
    
    def __init__(self):
        """Initialize parser with logging."""
        logger.debug(f"{self.__class__.__name__} initialized")
    
    def _validate_file(self, file_path: str | Path) -> Path:
        """
        Validate file path and return normalized Path object.
        
        Args:
            file_path: Path to file (string or Path object)
            
        Returns:
            Resolved Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a file or file is empty
        """
        logger.debug(f"Validating file: {file_path}")
        
        path = Path(file_path).expanduser()
        
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            msg = f"File not found: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        if not path.exists():
            msg = f"File not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        if not path.is_file():
            msg = f"Path is not a file: {path}"
            logger.error(msg)
            raise ValueError(msg)
        
        if not path.stat().st_size:
            msg = f"File is empty: {path}"
            logger.error(msg)
            raise ValueError(msg)
        
        file_size_kb = path.stat().st_size / 1024
        logger.debug(f"File validated: {path} ({file_size_kb:.1f} KB)")
        
        return path
    
    @abstractmethod
    def parse(self, file_path: str | Path) -> str:
        """
        Extract raw text from a file.
        
        Must be implemented by subclasses.
        
        Args:
            file_path: Path to file (string or Path object)
            
        Returns:
            Extracted text as string
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement parse() method")
    
    def _log_parse_start(self, file_path: Path) -> None:
        """Log start of parsing operation."""
        logger.info(f"Parsing file: {file_path.name}")
    
    def _log_parse_end(self, file_path: Path, text_length: int) -> None:
        """Log end of parsing operation."""
        logger.info(f"Parsed {text_length} characters from {file_path.name}")
    
    def _log_parse_error(self, file_path: Path, error: str) -> None:
        """Log parsing error."""
        logger.error(f"Failed to parse {file_path.name}: {error}")
