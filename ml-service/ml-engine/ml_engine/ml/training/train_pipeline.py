# pyre-ignore-all-errors
"""
train_pipeline.py
=================
End-to-end training pipeline:
  1. Generate fresh synthetic dataset (2000 samples, 165 features)
  2. Load and split dataset (80/20 train-test)
  3. Train RandomForestModel
  4. Evaluate on held-out test set
  5. Print feature importances (top 15)
  6. Save model artifact

Run directly
------------
  # From ml-engine/ directory:
  ../../.venv/bin/python -m ml_engine.ml.training.train_pipeline

  # Or via run_test.sh (runs pipeline + training):
  bash run_test.sh --train
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from ..dataset.synthetic_dataset import SyntheticDatasetGenerator
from ..dataset.dataset_loader import DATASET_LOADER
from ..models.random_forest_model import RandomForestModel, LABEL_NAMES

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent.parent.parent          # ml-service/ml-engine/
_DATASET_PATH = _ML_ROOT / "ml_engine" / "ml" / "datasets" / "resume_dataset.csv"
_ARTIFACT_DIR = _ML_ROOT / "ml_engine" / "ml" / "artifacts"
_MODEL_FILE = _ARTIFACT_DIR / "resume_rf_model.joblib"

# ── Label names for reporting ──────────────────────────────────────────────────
_LABEL_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]


class TrainingPipelineError(Exception):
    pass


# =============================================================================
# Core training function
# =============================================================================

def run_training(
    n_samples: int = 2000,
    regenerate: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
) -> RandomForestModel:
    """
    Full train cycle.

    Parameters
    ----------
    n_samples    : number of synthetic resumes to generate
    regenerate   : if True (default), always regenerate fresh dataset
    test_size    : fraction of data held out for evaluation
    random_state : reproducibility seed

    Returns
    -------
    Trained RandomForestModel instance.
    """

    sep = "─" * 60
    print(f"\n{sep}")
    print("  🤖  Resume ML Training Pipeline")
    print(f"  Dataset  : {n_samples} samples, 165 features")
    print(f"  Model    : RandomForestClassifier (300 trees, balanced)")
    print(f"{sep}\n")

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    if regenerate or not _DATASET_PATH.exists():
        print("  [1/5] Generating synthetic dataset...")
        gen = SyntheticDatasetGenerator()
        gen.generate(n_samples)
    else:
        print(f"  [1/5] Reusing existing dataset: {_DATASET_PATH.name}")

    # ── Step 2: Load ──────────────────────────────────────────────────────────
    print("  [2/5] Loading dataset...")
    X_full, y_full = DATASET_LOADER.load(str(_DATASET_PATH))
    print(f"         Shape: {X_full.shape}  Labels: {dict(y_full.value_counts().sort_index())}")

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=test_size,
        stratify=y_full,
        random_state=random_state,
    )
    print(f"  [3/5] Split → train={len(X_train)}  test={len(X_test)}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print("  [4/5] Training RandomForestModel...")
    t0 = time.time()
    model = RandomForestModel()
    model.train(X_train, y_train)
    elapsed = time.time() - t0
    print(f"         Done in {elapsed:.1f}s")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print(f"\n  [5/5] Evaluation on held-out test set ({len(X_test)} samples)\n")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc*100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=_LABEL_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    header = f"  {'':20s}  " + "  ".join(f"{n[:8]:>8s}" for n in _LABEL_NAMES)
    print(header)
    for i, row_vals in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row_vals)
        print(f"  {_LABEL_NAMES[i][:20]:20s}  {row_str}")

    # Feature importance
    print(f"\n  Top 15 Most Important Features:")
    for rank, (feat, imp) in enumerate(model.feature_importances(top_n=15), 1):
        bar = "█" * int(imp * 200)
        print(f"  {rank:>2}. {feat:40s}  {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(_MODEL_FILE)
    print(f"\n  ✅  Model saved → {_MODEL_FILE}")
    print(f"{sep}\n")

    return model


# =============================================================================
# CLI entrypoint
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_training(n_samples=2000, regenerate=True)