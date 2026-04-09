"""
Canonical feature schema for IT screening.
"""

from __future__ import annotations

from typing import List


class FeatureSchemaError(Exception):
    pass


class FeatureSchema:
    def __init__(self, features: List[str]) -> None:
        if not features:
            raise FeatureSchemaError("Feature list cannot be empty")
        if len(set(features)) != len(features):
            raise FeatureSchemaError("Duplicate features in schema")
        self._features: List[str] = list(features)

    def get_features(self) -> List[str]:
        return list(self._features)

    def size(self) -> int:
        return len(self._features)

    def validate(self, feature_dict: dict) -> None:
        if not isinstance(feature_dict, dict):
            raise FeatureSchemaError("Feature input must be a dictionary")

    def default_row(self) -> dict:
        return {f: 0 for f in self._features}

    def align(self, raw: dict) -> dict:
        return {f: (raw.get(f) if raw.get(f) is not None else 0) for f in self._features}


FEATURE_LIST: List[str] = [
    "skills_count",
    "programming_languages_count",
    "framework_count",
    "database_count",
    "years_of_experience",
    "projects_count",
    "has_projects",
    "has_experience",
    "has_internship",
    "degree_type",
    "is_it_candidate",
    "score",
    "has_valid_contact",
    "typo_count",
    "ats_total_penalty_score",
    "overall_profile_strength",
    "quantified_impact_count",
    "online_presence_count",
]

FEATURE_SCHEMA = FeatureSchema(FEATURE_LIST)
