"""
Scoring modules for the Resume Intelligence Platform.

Provides rule-based ATS scoring for ranking resumes.
"""

from .ats_scorer import score_resume

__all__ = [
    "score_resume",
]