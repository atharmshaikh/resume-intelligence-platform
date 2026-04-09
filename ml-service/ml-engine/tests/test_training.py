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

from ml_engine.ml.pipelines.training import run_training
from ml_engine.ml.engine.random_forest import RandomForestModel

def test_training_pipeline_end_to_end() -> None:
    """
    Test generating data and training the RF model with 165 features.
    """
    # 1. Run pipeline (fast mode, 100 samples)
    # Using a dummy model_id for test separation
    model = run_training(n_samples=100, regenerate=True, test_size=0.2)
    
    # 2. Assert model trained
    assert isinstance(model, RandomForestModel), "Pipeline must return the model instance."
    assert model._trained is True, "Model must be marked as trained."
    assert len(model._feature_names) == 165, f"Expected 165 model features, got {len(model._feature_names)}"
    
    # 3. Assert artifact exists
    # Note: run_training uses the active_model_id from config
    model_id = model.metadata.model_id
    artifact_path = Path(model.save_path) if hasattr(model, 'save_path') else None
    
    # Fallback check if save_path not stored
    if not artifact_path:
        _HERE = Path(__file__).resolve().parent
        _ROOT = _HERE.parent
        artifact_path = _ROOT / "ml_engine" / "ml" / "artifacts" / f"{model_id}.joblib"
        
    assert artifact_path.exists(), f"Model artifact not found at {artifact_path}"
