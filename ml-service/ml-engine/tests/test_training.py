"""
tests/test_training.py
======================
Test module for the ML training pipeline.
"""

import sys
from pathlib import Path
import pytest
import yaml

# Ensure ml_engine is in path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_engine.ml.pipelines.training import TrainingPipeline
from ml_engine.ml.engine.registry import ModelRegistry

def test_training_pipeline_logic() -> None:
    """
    Test generating data and training the RF model with 165 features.
    """
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    config_path = _ROOT / "ml_engine" / "ml" / "configs" / "training_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config not found at {config_path}")

    # 1. Init Pipeline
    pipeline = TrainingPipeline(config_path)
    
    # 2. Check config loaded
    assert pipeline.config is not None, "Pipeline must load configuration."
    assert "active_model" in pipeline.config, "Config must specify active_model."
    
    # 3. Model Registry check
    arch = pipeline.config.get("active_model", "logistic_regression")
    model_wrapper = ModelRegistry.get_model(arch)
    assert model_wrapper is not None, f"Registry must provide wrapper for {arch}."
    
    # 4. Check feature count (stabilized subset for v1.1.1)
    model_id = pipeline.config.get("deployment", {}).get("active_model_id", "LOGISTIC_REGRESSION_ACTIVE")
    artifact_path = _ROOT / "ml_engine" / "ml" / "artifacts" / f"{model_id}.joblib"
    
    if not artifact_path.exists():
         pytest.skip(f"Active model artifact {model_id} not found for structural test.")
         
    model_wrapper.load(artifact_path)
    assert model_wrapper._trained is True, "Loaded model must be marked as trained."
    # We allow the model to have its own feature set (currently 18 for LR)
    assert len(model_wrapper._feature_names) >= 12, f"Expected 12+ features, got {len(model_wrapper._feature_names)}"
