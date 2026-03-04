"""
Data schema representing the normalized ATS structure.
Using a simple class keeps memory overhead low.
"""

from typing import Dict, List


class ResumeSchema:
    """
    Standard internal representation of a parsed resume.
    """

    def __init__(self):
        self.raw_text: str = ""
        self.sections: Dict[str, str] = {}

        self.skills: List[str] = []
        self.education: List[str] = []
        self.experience: List[str] = []

    def to_dict(self) -> dict:
        """
        Convert schema to dictionary for JSON/API usage.
        """
        return {
            "raw_text": self.raw_text,
            "sections": self.sections,
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,
        }