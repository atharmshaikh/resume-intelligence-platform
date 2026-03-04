"""
PDF parser implementation using pdfminer.
"""

from pdfminer.high_level import extract_text
from .base_parser import BaseParser


class PDFParser(BaseParser):

    def parse(self, file_path: str) -> str:
        try:
            text = extract_text(file_path)

            if not text or not text.strip():
                raise ValueError("PDF parsing returned empty content")

            return text

        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF: {file_path}") from exc