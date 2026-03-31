"""
Training Configuration
"""

from pathlib import Path


class TrainingConfig:

    DATASET_PATH = Path("ml_engine/ml/datasets/resume_dataset.csv")

    ARTIFACT_DIR = Path("ml-engine/ml/artifacts")

    MODEL_NAME = "random_forest"

    MODEL_FILE = "resume_model.pkl"

    RANDOM_STATE = 42


CONFIG = TrainingConfig()