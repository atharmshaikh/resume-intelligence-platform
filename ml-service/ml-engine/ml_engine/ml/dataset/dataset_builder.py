"""
Dataset Builder

Responsible for building dataset rows from raw feature dictionaries.
"""

from typing import Dict, List

from ..feature_store.feature_vector import FEATURE_VECTOR_BUILDER


class DatasetBuilderError(Exception):
    pass


class DatasetBuilder:

    def __init__(self):

        self.vector_builder = FEATURE_VECTOR_BUILDER

    # -----------------------------------------------------

    def build_row(self, raw_features: Dict, label: int | None = None) -> Dict:
        """
        Convert raw features into dataset row.
        """

        if not isinstance(raw_features, dict):
            raise DatasetBuilderError("raw_features must be a dictionary")

        row = self.vector_builder.to_dataset_row(raw_features, label)

        return row

    # -----------------------------------------------------

    def build_rows(
        self,
        features_list: List[Dict],
        labels: List[int] | None = None
    ) -> List[Dict]:
        """
        Build dataset rows for multiple samples.
        """

        rows = []

        for i, features in enumerate(features_list):

            label = None

            if labels:
                label = labels[i]

            row = self.build_row(features, label)

            rows.append(row)

        return rows


DATASET_BUILDER = DatasetBuilder()