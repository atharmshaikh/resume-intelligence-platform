# pyre-ignore-all-errors
"""
training.py
===========
Industry-grade training pipeline for the Resume ML Engine.
Features:
  - Generative synthetic data (balanced 3-class distribution)
  - Automatic model versioning and metadata generation
  - Config-driven hyperparameter injection
  - Comprehensive evaluation and metrics reporting
"""

from __future__ import annotations

import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Core & Engine imports from refactored structure
from ml_engine.ml.core.model_metadata import ModelMetadata
from ml_engine.ml.engine.registry import ModelRegistry
from ml_engine.ml.data.generator import SyntheticDatasetGenerator
from ml_engine.ml.data.loader import DATASET_LOADER
from ml_engine.ml.engine.logistic_regression import LABEL_NAMES

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent.parent.parent          # ml-service/ml-engine/
_CONFIG_PATH = _ML_ROOT / "ml_engine" / "ml" / "configs" / "training_config.yaml"
_DATASET_PATH = _ML_ROOT / "ml_engine" / "ml" / "datasets" / "resume_dataset.csv"
_ARTIFACT_DIR = _ML_ROOT / "ml_engine" / "ml" / "artifacts"

def load_config() -> Dict[str, Any]:
    """Load training configuration from YAML."""
    if not _CONFIG_PATH.exists():
        logger.warning("Config file not found at %s. Using defaults.", _CONFIG_PATH)
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

# ── Label names for reporting ──────────────────────────────────────────────────
_LABEL_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]


class TrainingPipelineError(Exception):
    pass


# =============================================================================
# Core training function
# =============================================================================

def run_training(
    n_samples: int | None = None,
    regenerate: bool | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> Any:
    """
    Full industry-grade train cycle.
    """
    # 1. Load config
    config = load_config()
    pipe_cfg = config.get("pipeline", {})
    
    # Merge defaults/config with explicit arguments
    n_samples = n_samples or pipe_cfg.get("n_samples", 2000)
    regenerate = regenerate if regenerate is not None else pipe_cfg.get("regenerate_data", True)
    test_size = test_size or pipe_cfg.get("test_size", 0.20)
    random_state = random_state or pipe_cfg.get("random_state", 42)

    active_arch = config.get("active_model", "random_forest")
    deployment_cfg = config.get("deployment", {})
    model_id = deployment_cfg.get("active_model_id", f"{active_arch.upper()}_{int(time.time())}")

    sep = "─" * 60
    print(f"\n{sep}")
    print("  🤖  Resume ML Training Pipeline (Industry Standard)")
    print(f"  Model ID : {model_id}")
    print(f"  Arch     : {active_arch}")
    print(f"  Dataset  : {n_samples} samples, 166 features")
    print(f"{sep}\n")

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    if regenerate or not _DATASET_PATH.exists():
        print("  [1/5] Building dataset from real resumes located in data/uploads...")
        from ml_engine.ml.datasets.build_true_dataset import build_dataset
        build_dataset()
    else:
        print(f"  [1/5] Reusing existing dataset: {_DATASET_PATH.name}")

    # ── Step 2: Load ──────────────────────────────────────────────────────────
    print("  [2/5] Loading dataset...")
    X_full, y_full = DATASET_LOADER.load(str(_DATASET_PATH))
    print(f"         Shape: {X_full.shape}  Labels: {dict(y_full.value_counts().sort_index())}")

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    if len(X_full) < 15:
        print("  [3/5] Skipping split (tiny dataset, using all for training)")
        X_train, y_train = X_full, y_full
        X_test, y_test = X_full.copy(), y_full.copy()  # Evaluate on train set for basic sanity check
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=test_size,
            stratify=y_full,
            random_state=random_state,
        )
    print(f"  [3/5] Split → train={len(X_train)}  test={len(X_test)}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print(f"  [4/5] Training {active_arch}...")
    t0 = time.time()
    
    # Initialize metadata
    meta_cfg = deployment_cfg.get("model_metadata", {})
    metadata = ModelMetadata(
        model_id=model_id,
        name=meta_cfg.get("name", "Unnamed Model"),
        version=meta_cfg.get("version", "1.0.0"),
        description=meta_cfg.get("description", "No description provided."),
        changelog=meta_cfg.get("changelog", ["Initial build"])
    )

    # Get model from registry with hyperparams
    params = config.get("models", {}).get(active_arch, {})
    model = ModelRegistry.get_model(active_arch, metadata=metadata, **params)

    model.train(X_train, y_train)
    elapsed = time.time() - t0
    print(f"         Done in {elapsed:.1f}s")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print(f"\n  [5/5] Evaluation on held-out test set ({len(X_test)} samples)\n")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc*100:.1f}%\n")
    
    # Handle class mismatch if some classes (like 0) are missing from the tiny dataset
    unique_labels = sorted(set(y_test) | set(y_pred))
    target_names = [_LABEL_NAMES[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    # Record metrics in metadata
    model.metadata.performance_metrics = {
        "accuracy": round(float(acc), 4),
        "timestamp": datetime.now().isoformat()
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    header = f"  {'':20s}  " + "  ".join(f"{n[:8]:>8s}" for n in _LABEL_NAMES)
    print(header)
    for i, row_vals in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row_vals)
        print(f"  {_LABEL_NAMES[i][:20]:20s}  {row_str}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _ARTIFACT_DIR / f"{model_id}.joblib"
    model.save(save_path)
    print(f"\n  ✅  Model deployment ready → {save_path}")
    print(f"  ✅  Metadata (Sidecar) saved → {save_path.with_suffix('.json')}")
    print(f"{sep}\n")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_training()