# pyre-ignore-all-errors
"""
predictor.py
============
Inference module for Resume Pipeline.
Loads the trained Random Forest model and performs predictions on extracted features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..feature_store.feature_schema import FEATURE_SCHEMA
from ..feature_store.feature_vector import FEATURE_VECTOR_BUILDER
from ..models.random_forest_model import RandomForestModel

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    pass


class ResumePredictor:
    """Loads a trained RandomForestModel and predicts candidate strength."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise PredictorError(
                f"Model artifact not found at: {self.model_path}. "
                f"Run training pipeline first."
            )

        # Load trained RandomForestModel
        self.model = RandomForestModel()
        try:
            self.model.load(self.model_path)
            logger.info("Successfully loaded model from %s", self.model_path.name)
        except Exception as e:
            raise PredictorError(f"Failed to load model from {self.model_path}: {e}")

    # --------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw feature dictionary (from feature_extractor),
        aligns it to the 165-feature schema, and predicts.

        Returns
        -------
        dict:
            "label": int (0: Weak, 1: Average, 2: Strong)
            "label_name": str ("Weak Candidate"...)
            "confidence": float (0.0 - 1.0)
            "class_probs": dict {0: p0, 1: p1, 2: p2}
            "readiness": float (candidate_readiness_score)
        """
        if not isinstance(features, dict):
            raise PredictorError("Features must be a dictionary")

        # 1. Soft-validate
        FEATURE_SCHEMA.validate(features)

        # 2. Convert to ordered ML vector list
        # Missing values → 0
        feature_list = FEATURE_VECTOR_BUILDER.to_vector(features)

        # 3. Create DataFrame for sklearn
        feature_names = FEATURE_SCHEMA.get_features()
        df = pd.DataFrame([feature_list], columns=feature_names)

        # 4. We can safely overwrite the `candidate_readiness_score` column
        # with the actual input value so `predict_single` can echo it back.
        if "candidate_readiness_score" in features:
            df["candidate_readiness_score"] = float(features["candidate_readiness_score"])

        # 5. Predict using the RF model's structured output method
        result = self.model.predict_single(df)
        
        return result