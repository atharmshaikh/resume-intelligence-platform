"""
PDF parser implementation using pdfminer.
"""

from pdfminer.high_level import extract_text
from .base_parser import BaseParser
from pathlib import Path
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.layout import LAParams

class PDFParser(BaseParser):

    def parse(self, file_path: str) -> str:
        try:
            path = self._validate_file(file_path)
            
            laparams = LAParams(
            line_margin=0.4,
            word_margin=0.1,
            char_margin=2.0,
            )

            text = extract_text(
                str(path),
                laparams=laparams
            )

            # Safety guard: prevent extremely large PDF text
            if text and len(text) > 2_000_000:
                raise RuntimeError("PDF text exceeds safe processing size")

            if not text or len(text.strip().split()) < 10:
                raise ValueError("PDF parsing returned empty content")

            text = text.replace("\r", "\n")

            # collapse excessive blank lines
            text = "\n".join(line for line in text.splitlines())

            return text

        except PDFSyntaxError as exc:
            raise RuntimeError(f"Invalid or corrupted PDF file: {file_path}") from exc

        except Exception as exc:
            raise RuntimeError(
                f"PDF parsing failed for file: {file_path}"
            ) from exc