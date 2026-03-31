"""
Centralized Exception Catalog.

Defines all custom errors that can be raised during
the resume pipeline execution to prevent blank crashes.
"""

class ResumeEngineError(Exception):
    """Base exception for all Resume Engine errors."""
    pass

class ResumeParserError(ResumeEngineError):
    """Raised when a document cannot be parsed (corrupted, empty, or unsupported)."""
    pass

class ExtractionError(ResumeEngineError):
    """Raised when critical entity extraction fails unexpectedly."""
    pass

class PipelineTimeoutError(ResumeEngineError):
    """Raised when the processing of a resume exceeds the safe time limit (Anti-block)."""
    pass
