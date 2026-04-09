"""
logistic_regression.py
======================
Industry-grade Logistic Regression model for resume ATS ranking.
Part of the refined ml_engine.ml.engine package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from ..core.base_model import BaseModel
from ..core.model_metadata import ModelMetadata

logger = logging.getLogger(__name__)

LABEL_NAMES: Dict[int, str] = {
    0: "Weak Candidate",
    1: "Average Candidate",
    2: "Strong Candidate",
}


class LogisticRegressionError(Exception):
    """Raised on model-level errors."""
    pass


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier for 3-class resume ranking.
    Supports industry-standard versioning and metadata manifests.
    """

    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        super().__init__(metadata)
        self.model: LogisticRegression = LogisticRegression(
            max_iter=1000, multi_class="multinomial", class_weight="balanced"
        )
        self._feature_names: List[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._trained: bool = False

    # ──────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        if X.empty:
            raise LogisticRegressionError("Training data X is empty")

        self._feature_names = list(X.columns)
        self.model.fit(X, y)  # type: ignore[union-attr]
        self._trained = True

        if self.metadata:
            self.metadata.features_count = len(self._feature_names)

    # ──────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
        """Return predicted class labels."""
        self._check_trained()
        return self.model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix (n_samples × 3 classes)."""
        self._check_trained()
        return self.model.predict_proba(X)  # type: ignore[union-attr]

    def predict_single(self, feature_vector: pd.DataFrame) -> Dict[str, Any]:
        """Predict a single resume and return a structured result dict."""
        self._check_trained()

        label_int: int = int(self.predict(feature_vector)[0])
        proba: np.ndarray = self.predict_proba(feature_vector)[0]

        class_probs = {i: round(float(p), 4) for i, p in enumerate(proba)}

        heuristic_readiness = 0.0
        if "candidate_readiness_score" in feature_vector.columns:
            heuristic_readiness = float(feature_vector["candidate_readiness_score"].iloc[0])
        readiness = round(float(proba[1]) * 55.0 + float(proba[2]) * 100.0, 2)

        return {
            "label":       label_int,
            "label_name":  LABEL_NAMES.get(label_int, "Unknown"),
            "confidence":  round(float(proba[label_int]), 4),
            "class_probs": class_probs,
            "readiness":   readiness,
            "heuristic_readiness": round(heuristic_readiness, 2),
            "model_id":    self.metadata.model_id if self.metadata else "unknown"
        }

    # ──────────────────────────────────────────────────────
    # Feature importance
    # ──────────────────────────────────────────────────────

    def feature_importances(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Return top-N features by importance (using absolute coefficients)."""
        self._check_trained()
        if not self._feature_names:
            return []

        # For multinomial, use mean absolute coefficient across classes
        coef: np.ndarray = self.model.coef_  # type: ignore[union-attr]
        importances: np.ndarray = np.mean(np.abs(coef), axis=0)

        pairs = sorted(
            zip(self._feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:top_n]

    def explain_single(self, feature_vector: pd.DataFrame, target_class: int = 2, top_n: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Explain the prediction mathematically for XAI justifications."""
        self._check_trained()
        if not self._feature_names or self.model is None:
            return {"supports": [], "detractors": []}
            
        classes = self.model.classes_
        if len(classes) == 2:
            # Binary case: coef_[0] corresponds to the higher class label
            # e.g. if classes are [1, 2], then coef_[0] represents positive pull towards class 2
            coef = self.model.coef_[0]
            if target_class == classes[0]: # Lower class
                coef = -coef
        else:
            # Multiclass case: coef_[idx] corresponds to target_class index in model.classes_
            class_idx = np.where(classes == target_class)[0][0]
            coef = self.model.coef_[class_idx]
            
        vector = feature_vector.iloc[0].values
        contributions = coef * vector
        
        pairs = list(zip(self._feature_names, contributions.tolist()))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        supports = [p for p in sorted_pairs if p[1] > 0][:top_n]
        detractors = [p for p in reversed(sorted_pairs) if p[1] < 0][:top_n]
        
        return {
            "supports": supports,
            "detractors": detractors
        }

    # ──────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize model + metadata to disk."""
        self._check_trained()
        path = Path(path)

        # Save model artifact
        payload = {
            "model":         self.model,
            "feature_names": self._feature_names,
            "metadata":      self.metadata
        }
        joblib.dump(payload, path)

        # Also save side-car metadata JSON for transparency
        if self.metadata:
            meta_path = path.with_suffix(".json")
            self.metadata.to_json(meta_path)
            logger.info("Saved model metadata to %s", meta_path)

    def load(self, path: str | Path) -> None:
        """Deserialize model + metadata from disk."""
        path = Path(path)
        if not path.exists():
            raise LogisticRegressionError(f"Model file not found: {path}")

        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self._feature_names = payload.get("feature_names", [])
            self.metadata = payload.get("metadata")
        else:
            # Legacy support or direct model file
            self.model = payload  # type: ignore

        self._trained = True

    # ──────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._trained or self.model is None:
            raise LogisticRegressionError(
                "Model is not trained. Call train() or load() first."
            )
