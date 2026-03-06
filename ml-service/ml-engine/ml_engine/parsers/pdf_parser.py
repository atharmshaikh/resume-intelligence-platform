"""
PDF parser implementation using pdfminer.
"""

from pdfminer.high_level import extract_text
from .base_parser import BaseParser
from pathlib import Path
from pdfminer.pdfparser import PDFSyntaxError

class PDFParser(BaseParser):

    def parse(self, file_path: str) -> str:
        try:
            path = self._validate_file(file_path)
            text = extract_text(str(path))

            if not text or len(text.strip()) < 30:
                raise ValueError("PDF parsing returned empty content")

            return text

        except PDFSyntaxError as exc:
            raise RuntimeError(f"Invalid or corrupted PDF file: {file_path}") from exc

        except Exception as exc:
            raise RuntimeError(
                f"PDF parsing failed for file: {file_path}"
            ) from exc