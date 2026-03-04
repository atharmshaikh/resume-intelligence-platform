"""
Base interface for all document parsers.
Ensures consistent parser behavior across file types.
"""

from abc import ABC, abstractmethod


class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """
        Extract raw text from a file.
        Must be implemented by subclasses.
        """
        raise NotImplementedError