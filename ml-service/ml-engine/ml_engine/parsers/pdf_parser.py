"""
PDF Parser with 2-Column Layout Recovery

Extracts text from PDF files with robust handling for:
- Single-column layouts
- Two-column layouts
- Mixed layouts
- Project recovery from garbled text
"""

import logging
import re
from pathlib import Path
from typing import List, Set

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.layout import LAParams

from .base_parser import BaseParser
from ml_engine.utils import ResumeParserError
from .layout_parser import normalize_layout

logger = logging.getLogger(__name__)

# Recovery parser patterns
_BULLET_RE = re.compile(r'^\s*[-•●▪◦►▸■□◆◇*]\s*(.+)$')
_COLON_SPLIT_RE = re.compile(r'^([^:]+):\s*(.+)$')
_ACTION_VERBS = {"developed", "built", "created", "made", "implemented", "designed", "engineered"}
_PROJECT_KEYWORDS = {"project", "portfolio", "application", "system", "platform", "website", "app"}


class PDFParser(BaseParser):
    """
    PDF parser with 2-column layout handling.
    
    Features:
    - Standard PDF text extraction
    - 2-column layout detection and normalization
    - Project recovery for garbled text
    """
    
    def __init__(self):
        """Initialize PDF parser."""
        super().__init__()
        logger.debug("PDFParser initialized")
    
    def parse(self, file_path: str | Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text (layout-optimized)
        """
        try:
            path = self._validate_file(file_path)
            self._log_parse_start(path)
            
            # Step 1: Extract block-level data via PyMuPDF (Superior for Layout)
            import fitz
            doc = fitz.open(str(path))
            
            full_text = []
            is_two_column_doc = False
            
            for page in doc:
                blocks = page.get_text("blocks")
                # Sort blocks: Primary by y-coord, Secondary by x-coord
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                mid_x = page.rect.width / 2
                left_col = []
                right_col = []
                
                # Heuristic: if many blocks are clearly divided by the vertical center
                centered_gaps = 0
                for b in blocks:
                    if b[2] < mid_x:
                        left_col.append(b[4])
                    elif b[0] > mid_x:
                        right_col.append(b[4])
                    else:
                        # Spans both? Probably a header
                        left_col.append(b[4])
                
                if len(left_col) > 4 and len(right_col) > 4:
                    is_two_column_doc = True
                    full_text.extend(left_col)
                    full_text.extend(right_col)
                else:
                    # Single column or header-heavy: use block order
                    full_text.extend([b[4] for b in blocks])
            
            doc.close()
            text = "\n".join(full_text)
            
            # Step 2: Fallback to pdfminer if content is extremely sparse
            if len(text.strip().split()) < 20:
                logger.info("PyMuPDF yield low content, falling back to pdfminer...")
                laparams = LAParams(boxes_flow=0.5, all_texts=True)
                text = extract_text(str(path), laparams=laparams)
            
            # Step 3: Project Recovery (Crucial for 2-column)
            recovered_projects = self._extract_projects_recovery(text)
            if recovered_projects:
                text += "\n\n[RECOVERED_PROJECTS]\n" + "\n".join(recovered_projects)
            
            # Normalize and return
            text = text.replace("\r", "\n")
            self._log_parse_end(path, len(text))
            return text
            
        except PDFSyntaxError as exc:
            msg = f"Invalid or corrupted PDF file: {file_path}"
            self._log_parse_error(Path(str(file_path)), msg)
            raise ResumeParserError(msg) from exc
            
        except ResumeParserError:
            raise
            
        except ValueError as exc:
            if "empty" in str(exc).lower():
                logger.warning(f"Skipping empty PDF file: {file_path}")
            raise ResumeParserError(str(exc)) from exc
            
        except Exception as exc:
            msg = f"Unexpected PDF parsing failure: {file_path}"
            self._log_parse_error(Path(str(file_path)), msg)
            raise ResumeParserError(msg) from exc
    
    def _extract_projects_recovery(self, raw_text: str) -> List[str]:
        """
        Recovery parser for projects missed due to 2-column layout issues.
        
        Scans raw text line by line for project patterns:
        - Lines with colons (Name: description format)
        - Lines starting with bullets
        - Lines containing action verbs + project keywords
        
        Args:
            raw_text: Raw text from PDF
            
        Returns:
            List of recovered project entries
        """
        recovered: List[str] = []
        lines = raw_text.splitlines()
        seen_names: Set[str] = set()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if len(line) < 5:
                i += 1
                continue
            
            line_lower = line.lower()
            
            # Skip obvious non-project lines
            if any(skip in line_lower for skip in (
                "skills", "education", "experience", "languages",
                "interests", "hobbies", "certifications", "objective",
                "summary", "profile", "contact", "phone", "email"
            )):
                i += 1
                continue
            
            is_project_line = False
            project_name = ""
            project_desc = ""
            
            # Pattern 1: Colon format "Project Name: description"
            colon_match = _COLON_SPLIT_RE.match(line)
            if colon_match:
                potential_name = colon_match.group(1).strip()
                potential_desc = colon_match.group(2).strip()
                
                if (2 <= len(potential_name.split()) <= 5 and 
                    len(potential_desc.split()) >= 3 and
                    potential_name.lower() not in seen_names):
                    is_project_line = True
                    project_name = potential_name
                    project_desc = potential_desc
            
            # Pattern 2: Bullet point with project content
            bullet_match = _BULLET_RE.match(line)
            if bullet_match and not is_project_line:
                bullet_content = bullet_match.group(1).strip()
                
                if (any(v in bullet_content.lower() for v in _ACTION_VERBS) or
                    any(k in bullet_content.lower() for k in _PROJECT_KEYWORDS)):
                    inner_colon = _COLON_SPLIT_RE.match(bullet_content)
                    if inner_colon:
                        project_name = inner_colon.group(1).strip()
                        project_desc = inner_colon.group(2).strip()
                        if len(project_name.split()) <= 5 and project_name.lower() not in seen_names:
                            is_project_line = True
                    else:
                        words = bullet_content.split()
                        if len(words) >= 6:
                            project_name = " ".join(words[:4])
                            project_desc = " ".join(words[4:])
                            if project_name.lower() not in seen_names:
                                is_project_line = True
            
            # Pattern 3: Line with action verb + project keyword
            if not is_project_line:
                has_action = any(v in line_lower for v in _ACTION_VERBS)
                has_project_kw = any(k in line_lower for k in _PROJECT_KEYWORDS)
                if has_action and has_project_kw:
                    words = line.split()
                    if len(words) >= 5:
                        project_name = " ".join(words[:4])
                        project_desc = " ".join(words[4:])
                        if project_name.lower() not in seen_names:
                            is_project_line = True
            
            # Pattern 4: Check next line for continuation
            if not is_project_line and i + 1 < len(lines):
                next_line = lines[i + 1].strip().lower()
                has_action_next = any(v in next_line for v in _ACTION_VERBS)
                if has_action_next and len(line.split()) <= 5:
                    project_name = line
                    project_desc = lines[i + 1].strip()
                    if project_name.lower() not in seen_names and len(project_desc.split()) >= 4:
                        is_project_line = True
                        i += 1
            
            if is_project_line and project_name:
                project_name = re.sub(r'^[-•●▪◦►▸■□◆◇*\s]+', '', project_name).strip()
                project_name = re.sub(r'[:\s]+$', '', project_name).strip()
                
                if project_name.lower() in {
                    "projects", "project details", "other projects",
                    "academic projects", "personal projects"
                }:
                    i += 1
                    continue
                
                seen_names.add(project_name.lower())
                
                if project_desc:
                    recovered.append(f"{project_name}: {project_desc}")
                else:
                    recovered.append(project_name)
            
            i += 1
        
        return recovered
