"""
Block Segmenter - Resume Text Structure Analysis

Groups raw text lines into logical blocks:
- Headings (section titles)
- Paragraphs (continuous text)
- Bullet groups (lists)
- Contact info blocks

This enables smarter section detection than line-by-line parsing.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Block types
BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_BULLET_GROUP = "bullet_group"
BLOCK_CONTACT = "contact"
BLOCK_UNKNOWN = "unknown"

# Bullet patterns
_BULLET_PATTERN = re.compile(r'^\s*[•●▪◦►▸■□◆◇\-\*]\s+')
_BULLET_CHARS = {'•', '●', '▪', '◦', '►', '▸', '■', '□', '◆', '◇', '-', '*'}

# Heading indicators
_HEADING_INDICATORS = {
    'skills', 'skill', 'technical skills', 'technical expertise',
    'education', 'educational background', 'academic background',
    'experience', 'work experience', 'professional experience', 'employment',
    'projects', 'project', 'academic projects', 'personal projects',
    'achievements', 'awards', 'certifications', 'certificates',
    'languages', 'language proficiency',
    'interests', 'hobbies', 'activities',
    'summary', 'objective', 'career objective', 'profile',
    'contact', 'contact information', 'personal information',
}


@dataclass
class TextBlock:
    """Represents a logical block of text."""
    type: str
    content: List[str]
    start_line: int = 0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BlockSegmenter:
    """
    Segments resume text into logical blocks.
    
    Groups related lines together for better section detection.
    """
    
    def __init__(self):
        self.blocks: List[TextBlock] = []
    
    def segment(self, text: str) -> List[TextBlock]:
        """
        Segment text into logical blocks.
        
        Args:
            text: Cleaned resume text
            
        Returns:
            List of TextBlock objects
        """
        logger.info("Segmenting text into logical blocks...")
        
        lines = text.splitlines()
        self.blocks = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Try to identify block type
            if self._is_heading(line):
                block = self._extract_heading_block(lines, i)
                self.blocks.append(block)
                i = block.start_line + len(block.content)
            elif self._is_bullet_line(line):
                block = self._extract_bullet_group(lines, i)
                self.blocks.append(block)
                i = block.start_line + len(block.content)
            elif self._is_contact_line(line):
                block = self._extract_contact_block(lines, i)
                self.blocks.append(block)
                i = block.start_line + len(block.content)
            else:
                block = self._extract_paragraph(lines, i)
                self.blocks.append(block)
                i = block.start_line + len(block.content)
        
        logger.info(f"Segmented into {len(self.blocks)} blocks")
        return self.blocks
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is a section heading."""
        stripped = line.strip()
        
        if not stripped or len(stripped) > 60:
            return False
        
        # Check for common heading patterns
        lower = stripped.lower()
        
        # Exact match with known headings
        if lower in _HEADING_INDICATORS:
            return True
        
        # All caps, short
        if stripped.isupper() and len(stripped.split()) <= 4:
            return True
        
        # Title case, short, no ending punctuation
        if stripped.istitle() and len(stripped.split()) <= 3:
            if not stripped.endswith(('.', ',', ';', ':')):
                return True
        
        return False
    
    def _is_bullet_line(self, line: str) -> bool:
        """Check if line starts with a bullet."""
        stripped = line.strip()
        if not stripped:
            return False
        return stripped[0] in _BULLET_CHARS or bool(_BULLET_PATTERN.match(line))
    
    def _is_contact_line(self, line: str) -> bool:
        """Check if line looks like contact information."""
        stripped = line.strip()
        
        # Email pattern
        if re.search(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', stripped):
            return True
        
        # Phone pattern
        if re.search(r'(?:\+?\d[\d\s\-]{8,13}\d)', stripped):
            return True
        
        # URL pattern
        if re.search(r'(?:https?://|www\.)\S+', stripped):
            return True
        
        # LinkedIn/GitHub
        if 'linkedin.com' in stripped.lower() or 'github.com' in stripped.lower():
            return True
        
        return False
    
    def _extract_heading_block(self, lines: List[str], start_idx: int) -> TextBlock:
        """Extract heading block (heading + optional subtitle)."""
        content = [lines[start_idx].strip()]
        
        # Check if next line is a subtitle (short, no bullet)
        if start_idx + 1 < len(lines):
            next_line = lines[start_idx + 1].strip()
            if next_line and not self._is_bullet_line(next_line) and not self._is_heading(next_line):
                if len(next_line.split()) <= 8:  # Short subtitle
                    content.append(next_line)
        
        return TextBlock(
            type=BLOCK_HEADING,
            content=content,
            start_line=start_idx,
            confidence=0.95
        )
    
    def _extract_bullet_group(self, lines: List[str], start_idx: int) -> TextBlock:
        """Extract consecutive bullet lines as a group."""
        content = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if self._is_bullet_line(line):
                content.append(line)
                i += 1
            else:
                break
        
        return TextBlock(
            type=BLOCK_BULLET_GROUP,
            content=content,
            start_line=start_idx,
            confidence=0.9
        )
    
    def _extract_contact_block(self, lines: List[str], start_idx: int) -> TextBlock:
        """Extract contact information block."""
        content = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if self._is_contact_line(line):
                content.append(line)
                i += 1
            else:
                break
        
        return TextBlock(
            type=BLOCK_CONTACT,
            content=content,
            start_line=start_idx,
            confidence=0.85
        )
    
    def _extract_paragraph(self, lines: List[str], start_idx: int) -> TextBlock:
        """Extract paragraph (continuous non-bullet text)."""
        content = [lines[start_idx].strip()]
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                break
            
            if self._is_bullet_line(line) or self._is_heading(line):
                break
            
            content.append(line)
            i += 1
        
        return TextBlock(
            type=BLOCK_PARAGRAPH,
            content=content,
            start_line=start_idx,
            confidence=0.8
        )
    
    def get_blocks_by_type(self, block_type: str) -> List[TextBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.type == block_type]
    
    def get_headings(self) -> List[TextBlock]:
        """Get all heading blocks."""
        return self.get_blocks_by_type(BLOCK_HEADING)
    
    def get_bullet_groups(self) -> List[TextBlock]:
        """Get all bullet group blocks."""
        return self.get_blocks_by_type(BLOCK_BULLET_GROUP)


def segment_text(text: str) -> List[Dict[str, Any]]:
    """
    Convenience function to segment text.
    
    Args:
        text: Cleaned resume text
        
    Returns:
        List of block dictionaries
    """
    segmenter = BlockSegmenter()
    blocks = segmenter.segment(text)
    return [b.to_dict() for b in blocks]
