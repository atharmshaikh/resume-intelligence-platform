"""
Quality analysis modules for resume evaluation.

Includes:
- typo detection
- future resume quality checks
"""

from .typo_checker import count_typos, typo_score

__all__ = [
    "count_typos",
    "typo_score",
]