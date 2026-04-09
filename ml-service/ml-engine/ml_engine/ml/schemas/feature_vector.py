"""
Feature Vector Builder

Purpose
-------
Convert raw feature dictionaries into ML-ready vectors
using the canonical FEATURE_SCHEMA.

Responsibilities
----------------
- Enforce feature order
- Fill missing features safely
- Ignore unknown features
- Produce stable ML input vectors
"""

from typing import Dict, List, Any

from .feature_schema import FEATURE_SCHEMA


class FeatureVectorError(Exception):
    """Feature vector processing error."""
    pass


class FeatureVectorBuilder:
    """
    Builds ML feature vectors from raw feature dictionaries.
    """

    def __init__(self):

        self.schema = FEATURE_SCHEMA
        self.feature_list = self.schema.get_features()

    # -----------------------------------------------------
    # Build Ordered Feature Dict
    # -----------------------------------------------------

    def build_feature_dict(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Align raw features with schema.

        Missing features -> default 0
        Extra features -> ignored
        """

        if not isinstance(raw_features, dict):
            raise FeatureVectorError("raw_features must be a dictionary")

        aligned = {}

        for feature in self.feature_list:

            value = raw_features.get(feature, 0)

            # Normalize None values
            if value is None:
                value = 0

            aligned[feature] = value

        return aligned

    # -----------------------------------------------------
    # Convert to ML Vector
    # -----------------------------------------------------

    def to_vector(self, raw_features: Dict[str, Any]) -> List[Any]:
        """
        Convert feature dict into ordered list.
        """

        feature_dict = self.build_feature_dict(raw_features)

        return [feature_dict[f] for f in self.feature_list]

    # -----------------------------------------------------
    # Convert to Dataset Row
    # -----------------------------------------------------

    def to_dataset_row(
        self,
        raw_features: Dict[str, Any],
        label: int | None = None
    ) -> Dict[str, Any]:
        """
        Convert feature dict into dataset row.

        Optionally attach label.
        """

        row = self.build_feature_dict(raw_features)

        if label is not None:
            row["label"] = label

        return row

    # -----------------------------------------------------
    # Feature Count
    # -----------------------------------------------------

    def feature_count(self) -> int:
        """Return total number of features."""
        return len(self.feature_list)


# -----------------------------------------------------
# Global Builder Instance
# -----------------------------------------------------

FEATURE_VECTOR_BUILDER = FeatureVectorBuilder()