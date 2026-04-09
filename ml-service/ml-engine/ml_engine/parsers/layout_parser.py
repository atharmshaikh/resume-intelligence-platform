"""
Layout Parser - Two-Column Resume Handler

This module detects and normalizes multi-column resume layouts.
It converts two-column text into single flowing text while preserving
reading order and content integrity.

Key Features:
- Detects 2-column layout using spacing patterns
- Splits columns at whitespace gaps (not arbitrary midpoints)
- Reconstructs reading order: left column → right column
- Falls back to original text if detection is uncertain
"""

import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Constants for layout detection
MIN_LINE_LENGTH_FOR_DETECTION = 60  # Minimum line length to consider
MIN_GAP_WIDTH = 6  # Minimum consecutive spaces to be considered a column gap
MIN_LINES_FOR_DETECTION = 10  # Minimum lines needed to detect layout
COLUMN_DETECTION_THRESHOLD = 0.5  # Percentage of lines with gaps to trigger 2-column mode


class LayoutParser:
    """
    Parser for detecting and normalizing multi-column resume layouts.
    
    Converts two-column text into single flowing text.
    """
    
    def __init__(self):
        self.is_two_column = False
        self.column_gap_positions: List[int] = []
    
    def normalize(self, text: str) -> str:
        """
        Normalize resume layout to single-column format.
        
        Args:
            text: Raw text from PDF parser
            
        Returns:
            Text in single-column format if 2-column detected, else original
        """
        logger.info("Stage 2: Normalizing layout...")
        
        lines = text.splitlines()
        
        if len(lines) < MIN_LINES_FOR_DETECTION:
            logger.debug(f"Too few lines ({len(lines)}) for layout detection")
            return text
        
        # Detect layout
        self.is_two_column = self._detect_two_column_layout(lines)
        
        if self.is_two_column:
            logger.info("Detected 2-column structure → converting to single column...")
            return self._convert_to_single_column(lines)
        else:
            logger.debug("Single-column layout detected, no conversion needed")
            return text
    
    def _detect_two_column_layout(self, lines: List[str]) -> bool:
        """
        Detect if text has a two-column layout.
        
        Heuristics:
        1. Look for large whitespace gaps (≥6 spaces) in middle of lines
        2. Check if gap appears consistently across multiple lines
        3. Verify both sides of gap have meaningful content
        
        Returns:
            True if two-column layout detected
        """
        gap_count = 0
        total_valid_lines = 0
        
        gap_positions = []
        
        for line in lines[:100]:  # Sample first 100 lines
            if len(line) < MIN_LINE_LENGTH_FOR_DETECTION:
                continue
            
            total_valid_lines += 1
            
            # Look for large whitespace gap
            gap_match = re.search(r'\s{6,}', line)
            
            if gap_match:
                gap_start = gap_match.start()
                gap_end = gap_match.end()
                
                # Check if gap is in middle third of line (column separator)
                line_third = len(line) // 3
                if line_third <= gap_start <= 2 * line_third:
                    # Verify both sides have content
                    left_content = line[:gap_start].strip()
                    right_content = line[gap_end:].strip()
                    
                    if len(left_content) > 10 and len(right_content) > 10:
                        gap_count += 1
                        gap_positions.append(gap_start)
        
        if total_valid_lines < MIN_LINES_FOR_DETECTION:
            return False
        
        gap_ratio = gap_count / total_valid_lines
        
        logger.debug(f"Layout analysis: {gap_count}/{total_valid_lines} lines have column gaps ({gap_ratio:.1%})")
        
        # Store common gap positions for splitting
        if gap_ratio >= COLUMN_DETECTION_THRESHOLD:
            self.column_gap_positions = gap_positions
            return True
        
        return False
    
    def _convert_to_single_column(self, lines: List[str]) -> str:
        """
        Convert two-column text to single-column format.
        
        Strategy:
        1. Split each line at the column gap
        2. Collect all left-column content
        3. Collect all right-column content
        4. Join: left_column + "\\n\\n" + right_column
        
        Returns:
            Single-column text
        """
        left_column: List[str] = []
        right_column: List[str] = []
        
        # Find most common gap position
        if self.column_gap_positions:
            common_gap = max(set(self.column_gap_positions), key=self.column_gap_positions.count)
        else:
            common_gap = None
        
        for line in lines:
            if not line.strip():
                continue
            
            # Try to split at gap
            if common_gap and len(line) > common_gap:
                # Check if there's actually a gap at this position
                gap_match = re.search(r'\s{6,}', line[common_gap-3:common_gap+10])
                if gap_match:
                    actual_gap_start = common_gap - 3 + gap_match.start()
                    actual_gap_end = common_gap - 3 + gap_match.end()
                    
                    left_part = line[:actual_gap_start].strip()
                    right_part = line[actual_gap_end:].strip()
                    
                    if left_part:
                        left_column.append(left_part)
                    if right_part:
                        right_column.append(right_part)
                    continue
            
            # Fallback: search for gap in this specific line
            gap_match = re.search(r'\s{6,}', line)
            if gap_match and len(line) >= MIN_LINE_LENGTH_FOR_DETECTION:
                left_part = line[:gap_match.start()].strip()
                right_part = line[gap_match.end():].strip()
                
                if left_part:
                    left_column.append(left_part)
                if right_part:
                    right_column.append(right_part)
            else:
                # No gap found - line belongs to left column
                if line.strip():
                    left_column.append(line.strip())
        
        # Safety check: ensure we didn't lose content
        original_word_count = sum(len(line.split()) for line in lines)
        result_word_count = len(left_column) + len(right_column)
        
        if result_word_count < original_word_count * 0.8:
            logger.warning("Column conversion may have lost content, using fallback")
            return '\n'.join(line for line in lines if line.strip())
        
        # Combine columns
        left_text = '\n'.join(left_column)
        right_text = '\n'.join(right_column) if right_column else ''
        
        if right_text:
            result = left_text + '\n\n' + right_text
        else:
            result = left_text
        
        logger.info(f"Layout conversion complete: {len(left_column)} left + {len(right_column)} right column lines")
        
        return result


def normalize_layout(text: str) -> str:
    """
    Convenience function for layout normalization.
    
    Args:
        text: Raw text from PDF parser
        
    Returns:
        Normalized text (single-column if 2-column detected)
    """
    parser = LayoutParser()
    return parser.normalize(text)


def detect_layout_type(text: str) -> str:
    """
    Detect layout type of resume text.
    
    Args:
        text: Raw text
        
    Returns:
        'single-column' or 'two-column'
    """
    parser = LayoutParser()
    parser.normalize(text)  # Run detection
    return 'two-column' if parser.is_two_column else 'single-column'
