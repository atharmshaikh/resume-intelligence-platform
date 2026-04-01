"""
Pipeline package.

Contains orchestration logic that connects all modules of the
Resume Intelligence Platform.

Responsibilities:
- file validation
- parser selection
- resume processing workflow
"""

from .resume_pipeline import ResumePipeline

__all__ = [
    "ResumePipeline",
]