"""
DOCX parser using python-docx.
"""

from docx import Document
from .base_parser import BaseParser


class DocxParser(BaseParser):

    def parse(self, file_path: str) -> str:
        try:
            doc = Document(file_path)

            text_lines = []

            for paragraph in doc.paragraphs:
                content = paragraph.text.strip()
                if content:
                    text_lines.append(content)

            if not text_lines:
                raise ValueError("DOCX parsing returned empty content")

            return "\n".join(text_lines)

        except Exception as exc:
            raise RuntimeError(f"Failed to parse DOCX: {file_path}") from exc