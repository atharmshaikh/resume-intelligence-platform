"""
Data schema representing the normalized ATS structure.

This object is the central data container used across the pipeline.
It stores raw text, detected sections, normalized fields, and
candidate identity entities.

The design intentionally stays lightweight to avoid unnecessary
memory overhead during large-scale resume processing.
"""

from typing import Dict, List, Optional


class ResumeSchema:
    """
    Standard internal representation of a parsed resume.
    """

    def __init__(self):

        # -----------------------------
        # Raw resume data
        # -----------------------------

        self.raw_text: str = ""
        self.sections: Dict[str, str] = {}

        # -----------------------------
        # Candidate identity entities
        # (Stage-2 extraction)
        # -----------------------------

        self.name: Optional[str] = None
        self.email: Optional[str] = None
        self.phone: Optional[str] = None
        self.location: Optional[str] = None

        # -----------------------------
        # Structured ATS fields
        # -----------------------------

        self.skills: List[str] = []
        self.education: List[str] = []
        self.experience: List[str] = []

        self.features = {}

    def to_dict(self) -> dict:
        """
        Convert schema to dictionary for JSON/API usage.
        """

        return {

            # Identity
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,

            # ATS structured fields
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,

            # Raw parsing results
            "sections": self.sections,
            "raw_text": self.raw_text,

            "features": self.features,
        }