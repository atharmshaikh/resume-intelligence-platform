"""
Text cleaning utilities.
Keeps text normalization consistent across the engine.
"""

import re


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove redundant characters.
    """

    if not text:
        return ""

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()