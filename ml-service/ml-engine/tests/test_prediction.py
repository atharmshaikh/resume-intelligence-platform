"""
tests/test_prediction.py
========================
Predictor logic test with a dummy 18-feature vector (Logistic Regression baseline).
"""

import sys
import json
from pathlib import Path
import pytest

# Ensure ml_engine is in path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_engine.ml.inference.predictor import ResumePredictor
from ml_engine.ml.schemas.feature_schema import FEATURE_SCHEMA

def test_resume_predictor() -> None:
    """Predictor should return structured output with labels and probabilities."""
    model_path = _ROOT / "ml_engine" / "ml" / "artifacts" / "LOGISTIC_REGRESSION_ACTIVE.joblib"
    
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}, skipping predictor test.")

    # 1. Init Predictor
    predictor = ResumePredictor(model_path)

    # 2. Mock feature dict (18 features for Logistic Regression)
    # Give it good stats to force a Strong prediction (2)
    features = {
        "has_name": 1,
        "has_email": 1,
        "has_phone": 1,
        "candidate_readiness_score": 85.0, # High readiness
        "skills_count": 18,
        "skill_weight_score": 28.5,
        "has_projects": 1,
        "projects_count": 4,
        "has_experience": 1,
        "has_ml_skills": 1,
        "has_github": 1,
        "has_linkedin": 1,
        "has_cgpa": 1,
        "cgpa_value": 9.2,
        "is_cs_it_candidate": 1
    }
    
    # Complete the dict with zeroes so size matching the schema
    for f in FEATURE_SCHEMA.get_features():
        if f not in features:
            features[f] = 0.0

    # 3. Predict
    json_result = predictor.predict(features)
    
    # 4. Assert structure
    assert isinstance(json_result, str), "Predictor MUST return a JSON string"
    result = json.loads(json_result)
    
    assert isinstance(result, dict)
    assert "decision" in result
    assert "score" in result
    assert "reasons" in result
    assert "model_output" in result
    
    mp = result["model_output"]
    assert "label" in mp
    assert "label_name" in mp
    assert "confidence" in mp
    assert "class_probs" in mp
    assert "model_id" in mp
    assert "raw_label_name" in mp

    # 5. Assert values
    label = mp["label"]
    assert label in [0, 1, 2], f"Label must be 0, 1, or 2, got {label}"
    assert mp["class_probs"][str(label)] >= 0.33, "Highest probability class selected."
    assert 0.0 <= result["score"] <= 1.0, "Top-level score must be a probability (0-1)."
    assert result["decision"] in {"Rejected", "Manual Review", "Shortlisted"}
