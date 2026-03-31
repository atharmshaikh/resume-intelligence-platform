"""
tests/test_training.py
======================
Test module for the ML training pipeline.

We do a quick 'lite' run of the training pipeline here 
using fewer samples (n_samples=100) to keep CI fast.
"""

import sys
from pathlib import Path
import pytest

# Ensure ml_engine is in path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_engine.ml.training.train_pipeline import run_training
from ml_engine.ml.models.random_forest_model import RandomForestModel


def test_training_pipeline_end_to_end() -> None:
    """
    Test generating data and training the RF model with 165 features.
    """
    # 1. Run pipeline (fast mode, 100 samples)
    model = run_training(n_samples=100, regenerate=True, test_size=0.2)
    
    # 2. Assert model trained
    assert isinstance(model, RandomForestModel), "Pipeline must return the model instance."
    assert model._trained is True, "Model must be marked as trained."
    assert len(model._feature_names) == 165, f"Expected 165 features, got {len(model._feature_names)}"
    
    # 3. Assert artifact exists
    artifact_path = _ROOT / "ml_engine" / "ml" / "artifacts" / "resume_rf_model.joblib"
    assert artifact_path.exists(), f"Model artifact not found at {artifact_path}"