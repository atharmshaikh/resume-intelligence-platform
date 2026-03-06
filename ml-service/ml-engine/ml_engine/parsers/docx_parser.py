"""
DOCX parser using python-docx.
"""

from docx import Document
from .base_parser import BaseParser
from pathlib import Path
from docx.opc.exceptions import PackageNotFoundError

class DocxParser(BaseParser):

    def parse(self, file_path: str | Path) -> str:
        try:
            path = self._validate_file(file_path)
            doc = Document(str(path))   
            
            text_lines = []

            if len(text_lines) > 5000:
                raise RuntimeError("DOCX content too large to safely process")

            for paragraph in doc.paragraphs:
                content = paragraph.text.strip()
                if content:
                    text_lines.append(content)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " ".join(
                        cell.text.strip() 
                        for cell in row.cells 
                        if cell.text and cell.text.strip()
                    )
                    if row_text:
                        text_lines.append(row_text)

            if len(text_lines) < 3:
                raise ValueError("DOCX parsing returned empty content")

            text = "\n".join(text_lines)
            text = text.replace("\r", "\n")
            return text

        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Invalid or corrupted DOCX file: {file_path}"
        ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"DOCX parsing failed for file: {file_path}"
        ) from exc