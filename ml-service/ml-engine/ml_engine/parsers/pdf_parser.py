"""
PDF parser implementation using pdfminer.
"""

import logging
from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.layout import LAParams

from .base_parser import BaseParser
from ml_engine.utils import ResumeParserError

logger = logging.getLogger(__name__)

class PDFParser(BaseParser):

    def parse(self, file_path: str) -> str:
        try:
            path = self._validate_file(file_path)
            
            laparams = LAParams(
                boxes_flow=0.5,
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                all_texts=True,
            )

            text = extract_text(
                str(path),
                laparams=laparams
            )

            # Safety guard: prevent extremely large PDF text
            if text and len(text) > 2_000_000:
                logger.warning(f"PDF exceeds safe processing size: {file_path}")
                raise ResumeParserError("PDF text exceeds safe processing size")

            if not text or len(text.strip().split()) < 10:
                logger.warning(f"PDF parsing returned purely empty or sparse content: {file_path}")
                raise ResumeParserError("PDF parsing returned empty content")

            text = text.replace("\r", "\n")

            # collapse excessive blank lines
            text = "\n".join(line for line in text.splitlines())

            return text

        except PDFSyntaxError as exc:
            msg = f"Invalid or corrupted PDF file: {file_path}"
            logger.error(msg)
            raise ResumeParserError(msg) from exc

        except ResumeParserError:
            raise

        except Exception as exc:
            msg = f"Unexpected PDF parsing failure for file: {file_path}"
            logger.exception(msg)
            raise ResumeParserError(msg) from exc