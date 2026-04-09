"""
Data schema representing the normalized ATS structure.

This object is the central data container used across the pipeline.
It stores raw text, detected sections, normalized fields, and
candidate identity entities.

The design intentionally stays lightweight to avoid unnecessary
memory overhead during large-scale resume processing.
"""

from typing import Dict, List, Optional, Any


class ResumeSchema:
    """
    Standard internal representation of a parsed resume.
    """
    __slots__ = (
        "raw_text",
        "sections",

        "name",
        "email",
        "phone",
        "location",

        "skills",
        "education",
        "experience",
        "projects",
        "achievements",
        "certifications",
        "interests",
        "summary_lines",
        "languages",
        "education_details",
        "project_details",

        "features",
        "scores",
        "quality",
    )

    def __init__(self):

        # -----------------------------
        # Raw resume data
        # -----------------------------

        self.raw_text: str = ""
        self.sections: Dict[str, List[str]] = {}

        # -----------------------------
        # Candidate identity entities
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
        self.projects: List[str] = []

        # Optional resume sections
        self.achievements: List[str] = []
        self.certifications: List[str] = []
        self.interests: List[str] = []
        self.summary_lines: List[str] = []
        self.languages: List[str] = []
        self.education_details: List[Dict[str, Any]] = []
        self.project_details: List[Dict[str, Any]] = []
   
        # -----------------------------
        # Derived pipeline outputs
        # Populated during pipeline stages:
        # - feature extraction
        # - quality analysis
        # - ATS scoring 
        # -----------------------------
        self.features: Dict[str, Any] = {}
        self.scores: Dict[str, Any] = {}
        self.quality: Dict[str, Any] = {}

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
            "skills": list(self.skills),
            "education": list(self.education),
            "experience": list(self.experience),
            "projects": list(self.projects),

            "achievements": list(self.achievements),
            "certifications": list(self.certifications),
            "interests": list(self.interests),
            "summary": list(self.summary_lines),
            "languages": list(self.languages),
            "education_details": [dict(item) for item in self.education_details],
            "project_details": [dict(item) for item in self.project_details],

            # Raw parsing results
            "sections": {k: list(v) for k, v in self.sections.items()},
            "raw_text": self.raw_text,

            # Derived
            "features": dict(self.features),
            "scores": dict(self.scores),
            "quality": dict(self.quality),
        }
    
    def summary(self) -> Dict:
        """
        Lightweight debugging summary.
        """

        return {
            "name": self.name,
            "skills": len(self.skills),
            "education": len(self.education),
            "experience": len(self.experience),
            "projects": len(self.projects),
            "education_details": len(self.education_details),
            "project_details": len(self.project_details),
            "sections_detected": len(self.sections or {}),
        }
    
    def clear_derived(self) -> None:
        """
        Reset derived pipeline outputs.
        Useful when re-processing the same schema object.
        """
        self.features.clear()
        self.scores.clear()
        self.quality.clear()
