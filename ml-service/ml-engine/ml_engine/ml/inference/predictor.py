# pyre-ignore-all-errors
"""
predictor.py
============
Inference module for Resume Pipeline (Industry Grade).
- Config-driven model versioning and ID selection.
- Hard-rule pre-filtering (Mandatory contact info/skills).
- 3-Tier Decision logic (Shortlisted, Manual Review, Rejected).
"""

from __future__ import annotations

import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore

from ..schemas.feature_schema import FEATURE_SCHEMA
from ..schemas.feature_vector import FEATURE_VECTOR_BUILDER
from ..engine.registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    """Raised on inference failure."""
    pass


class ResumePredictor:
    """
    Industry-grade predictor for resume candidate ranking.
    Supports hard rules, config-driven thresholds, and versioned artifact loading.
    """

    def __init__(self, config_or_model_path: str | Path | None = None) -> None:
        _HERE = Path(__file__).resolve().parent
        _ML_ROOT = _HERE.parent.parent.parent
        self.artifacts_dir = _ML_ROOT / "ml_engine" / "ml" / "artifacts"
        self.model: Any = None
        self.wrapper: Any = None
        self._feature_names: List[str] = []

        # 1. Handle Initialization Overrides
        if config_or_model_path and str(config_or_model_path).endswith(('.joblib', '.pkl')):
            # Direct Model Override (Useful for testing)
            self.config_path = _ML_ROOT / "ml_engine" / "ml" / "configs" / "training_config.yaml"
            self.config = self._load_config()
            self.model_path = Path(config_or_model_path)
            self.model_id = self.model_path.stem
        else:
            # Standard Config-Based Initialization
            self.config_path = Path(config_or_model_path or (_ML_ROOT / "ml_engine" / "ml" / "configs" / "training_config.yaml"))
            self.config = self._load_config()
            self.deployment = self.config.get("deployment", {})
            self.model_id = self.deployment.get("active_model_id", "RANDOM_FOREST_V2026_04_01_PATCH_01")
            self.model_path = self.artifacts_dir / f"{self.model_id}.joblib"

        self.rules = self.config.get("inference_rules", {})

        if not self.model_path.exists():
            logger.warning(
                "Model artifact '%s' not found at: %s. "
                "Predictor will run in 'Parse-Only' mode until training is completed.",
                self.model_id, self.model_path
            )
            self.model = None
            self.wrapper = None
            self._feature_names = []
            return

        # Load model wrapper via registry if possible to support XAI
        try:
            # Determine architecture from model_id or config
            active_arch = self.config.get("active_model", "logistic_regression")
            if "RANDOM_FOREST" in self.model_id:
                active_arch = "random_forest"
            elif "XGBOOST" in self.model_id:
                active_arch = "xgboost"
            elif "LOGISTIC" in self.model_id:
                active_arch = "logistic_regression"
                
            self.wrapper = ModelRegistry.get_model(active_arch)
            self.wrapper.load(self.model_path)
            self._feature_names = getattr(self.wrapper, "_feature_names", [])
            logger.info("Successfully loaded active model wrapper: %s (%s)", self.model_id, active_arch)
        except Exception as e:
            logger.error("Failed to load model wrapper, falling back to raw joblib: %s", e)
            # Legacy fallback
            payload = joblib.load(self.model_path)
            if isinstance(payload, dict) and "model" in payload:
                self.model: Any = payload["model"]
                self._feature_names = payload.get("feature_names", [])
            else:
                self.model = payload
                self._feature_names = []
            self.wrapper = None

    def predict(self, features: Dict[str, Any]) -> str:
        """
        End-to-end inference including hard-rejection and 3-tier scoring.
        """
        if not isinstance(features, dict):
            raise PredictorError("Features must be a dictionary")

        reasons: List[str] = []

        # ── Step 1: Mandatory Hard Rules ──────────────────────────
        hard_reqs = self.rules.get("hard_requirements", {})

        # A. Contact completeness
        if hard_reqs.get("require_contact", True):
            # Support both old (has_email/has_phone) and new (has_valid_contact) schemas
            has_email = features.get("has_email", features.get("has_valid_contact", 0))
            has_phone = features.get("has_phone", features.get("has_valid_contact", 0))
            if not has_email or not has_phone:
                reasons.append("Missing mandatory contact information (Email/Phone)")

        # B. Minimum Projects
        min_projects = hard_reqs.get("min_projects", 0)
        if features.get("projects_count", 0) < min_projects:
            reasons.append(f"Insufficient projects (Found: {features.get('projects_count', 0)}, Required: {min_projects})")

        # C. Required Skills (Direct Reject if missing ANY)
        req_skills = hard_reqs.get("required_skills", [])

        # Check mapping to high-level feature results
        if "Machine Learning" in req_skills and not features.get("has_ai_ml_skills", 0):
            reasons.append("Missing required domain skill: Machine Learning")

        if "Web Development" in req_skills and not features.get("has_web_dev_skills", 0):
            reasons.append("Missing required domain skill: Web Development")

        # If any hard rule failed, return immediate rejection
        if reasons:
            return self._build_verdict(
                decision="Rejected",
                score=0,
                reasons=reasons,
                prediction=None
            )

        # ── Step 2: ML Scoring ──────────────────────────────────
        if self.model is None and self.wrapper is None:
            # Model is missing (Fresh Start mode)
            return self._build_verdict(
                decision="Pending Training",
                score=0.5,
                reasons=reasons + ["Model artifact missing; decision based on parser fallback."],
                prediction={
                    "label": -1,
                    "label_name": "Pending Training",
                    "confidence": 0,
                    "model_id": self.model_id
                }
            )

        FEATURE_SCHEMA.validate(features)
        feature_list = FEATURE_VECTOR_BUILDER.to_vector(features)
        feature_names = FEATURE_SCHEMA.get_features()
        df = pd.DataFrame([feature_list], columns=feature_names)
        
        # Use model's stored feature names if available
        model_features = self._feature_names or []
        if model_features:
            df = df.reindex(columns=model_features, fill_value=0)

        # Inference via wrapper or fallback
        if self.wrapper:
            proba = self.wrapper.predict_proba(df)[0]
            label_int = int(self.wrapper.predict(df)[0])
            classes = self.wrapper.model.classes_
        else:
            proba = self.model.predict_proba(df)[0]
            label_int = int(self.model.predict(df)[0])
            classes = getattr(self.model, "classes_", [])

        class_probs = {int(c): round(float(p), 4) for c, p in zip(classes, proba)}
        
        # Correctly map label_int to its index in the probability array
        class_idx = np.where(classes == label_int)[0][0]
        confidence = round(float(proba[class_idx]), 4)

        # XAI: Inject mathematical reasoning from the Logistic Regression coefficients
        if self.wrapper and hasattr(self.wrapper, "explain_single"):
            xai = self.wrapper.explain_single(df, target_class=label_int, top_n=2)
            
            for f_name, impact in xai.get("supports", []):
                val = features.get(f_name, 0)
                reasons.append(f"Strongly supported by '{f_name}' metric (Value: {val}).")
            
            for f_name, impact in xai.get("detractors", []):
                val = features.get(f_name, 0)
                reasons.append(f"Dragged down mathematically by '{f_name}' deficit (Value: {val}).")

        # Core Bug Fix: Ensure we score based on candidate "strength", not raw confidence.
        # Otherwise a 100% confident "Weak Candidate" prediction might get auto-Shortlisted!
        strength_score = class_probs.get(1, 0.0) * 0.5 + class_probs.get(2, 0.0) * 1.0
        score = strength_score

        # Map raw label to name (kept intact in model_output)
        label_map = {0: "Weak Candidate", 1: "Average Candidate", 2: "Strong Candidate"}
        raw_label_name = label_map.get(label_int, "Unknown")

        # ── Step 3: Tiered Threshold Logic ─────────────────────
        thresholds = self.rules.get("tier_thresholds", {"shortlisted": 0.65, "manual_review": 0.45})

        if strength_score >= thresholds.get("shortlisted", 0.65):
            decision = "Shortlisted"
            reasons.append(f"Profile strength ({strength_score:.2f}) easily exceeds shortlisted requirement.")
        elif strength_score >= thresholds.get("manual_review", 0.45):
            decision = "Manual Review"
            reasons.append(f"Model indicates partial fit ({strength_score:.2f} strength); requires human verification.")
        else:
            decision = "Rejected"
            reasons.append(f"Overall profile strength ({strength_score:.2f}) sits below actionable thresholds.")

        if confidence < 0.45:
            reasons.append(f"Warning: Low model classification confidence ({confidence:.2f}). Treat as uncertain.")

        # label_name matches the final decision, not raw model output
        prediction = {
            "label": label_int,
            "label_name": decision,
            "confidence": confidence,
            "class_probs": class_probs,
            "model_id": self.model_id,
            "raw_label_name": raw_label_name
        }

        return self._build_verdict(decision, score, reasons, prediction)

    def _build_verdict(self, decision: str, score: float, reasons: List[str], prediction: Optional[Dict]) -> str:
        """Create a consistent JSON payload."""
        payload = {
            "model_id": self.model_id,
            "decision": decision,
            "score": round(float(score), 2),
            "reasons": reasons,
            "model_output": prediction,
            "metadata": {
                "thresholds": self.rules.get("tier_thresholds"),
                "hard_rules_applied": True
            }
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
