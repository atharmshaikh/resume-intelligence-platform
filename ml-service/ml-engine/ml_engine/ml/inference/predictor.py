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

import pandas as pd

from ..schemas.feature_schema import FEATURE_SCHEMA
from ..schemas.feature_vector import FEATURE_VECTOR_BUILDER
from ..engine.random_forest import RandomForestModel

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
            raise PredictorError(
                f"Model artifact '{self.model_id}' not found at: {self.model_path}. "
                f"Please run training or update training_config.yaml."
            )

        self.model = RandomForestModel()
        try:
            self.model.load(self.model_path)
            logger.info("Successfully loaded active model: %s", self.model_id)
        except Exception as e:
            raise PredictorError(f"Failed to load model artifact {self.model_id}: {e}")

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
            if not features.get("has_email", 0) or not features.get("has_phone", 0):
                reasons.append("Missing mandatory contact information (Email/Phone)")
                
        # B. Minimum Projects
        min_projects = hard_reqs.get("min_projects", 0)
        if features.get("projects_count", 0) < min_projects:
            reasons.append(f"Insufficient projects (Found: {features.get('projects_count', 0)}, Required: {min_projects})")

        # C. Required Skills (Direct Reject if missing ANY)
        req_skills = hard_reqs.get("required_skills", [])
        # Note: In a real world app, we'd check raw extracted skills. 
        # Here we check flags derived from those skills (e.g., has_ml_skills).
        # We simulate the check using available flags.
        if "Machine Learning" in req_skills and not features.get("has_ml_skills", 0):
            reasons.append("Missing required domain skill: Machine Learning")

        # If any hard rule failed, return immediate rejection
        if reasons:
            return self._build_verdict(
                decision="Rejected",
                score=0,
                reasons=reasons,
                prediction=None
            )

        # ── Step 2: ML Scoring ──────────────────────────────────
        FEATURE_SCHEMA.validate(features)
        feature_list = FEATURE_VECTOR_BUILDER.to_vector(features)
        feature_names = FEATURE_SCHEMA.get_features()
        df = pd.DataFrame([feature_list], columns=feature_names)

        # Sync scores from extractor
        if "candidate_readiness_score" in features:
            df["candidate_readiness_score"] = float(features["candidate_readiness_score"])

        prediction = self.model.predict_single(df)
        
        # ── Step 3: Tiered Threshold Logic ─────────────────────
        score = prediction.get("readiness", features.get("candidate_readiness_score", 0))
        thresholds = self.rules.get("tier_thresholds", {"shortlisted": 90, "manual_review": 75})
        
        if score >= thresholds["shortlisted"]:
            decision = "Shortlisted"
        elif score >= thresholds["manual_review"]:
            decision = "Manual Review"
            reasons.append("Meets baseline criteria but requires human verification.")
        else:
            decision = "Rejected"
            reasons.append(f"Candidate strength score ({score:.1f}) below rejection threshold.")

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