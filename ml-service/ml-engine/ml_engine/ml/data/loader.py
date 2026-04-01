# pyre-ignore-all-errors
"""
loader.py
=========
Dataset loading and processing for training.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised on dataset loading failure."""
    pass


class DatasetLoader:
    """
    Handles CSV dataset ingestion.
    Separates features (X) from the target label (y).
    """

    def load(self, csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load CSV and return (X, y).
        """
        try:
            df = pd.read_csv(csv_path)
            
            if "label" not in df.columns:
                raise LoaderError(f"Missing 'label' column in {csv_path}")

            y = df["label"]
            X = df.drop(columns=["label"])

            return X, y

        except FileNotFoundError:
            raise LoaderError(f"Dataset not found: {csv_path}")
        except Exception as exc:
            raise LoaderError(f"Error loading dataset: {exc}")


DATASET_LOADER = DatasetLoader()