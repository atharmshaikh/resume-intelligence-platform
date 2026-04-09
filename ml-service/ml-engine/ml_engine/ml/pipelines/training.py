# pyre-ignore-all-errors
"""
training.py
===========
Core training orchestration pipeline for the Resume ML Engine.
- Loads declarative rules from YAML.
- Ingests processed JSON data.
- Trains the selected model architecture.
- Deploys versioned artifacts.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict

from ml_engine.ml.data.loader import DATASET_LOADER
from ml_engine.ml.engine.registry import ModelRegistry
from ml_engine.ml.core.model_metadata import ModelMetadata

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the fresh training cycle from data to deployment.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.runtime = self.config.get("runtime", {})
        
        # Robust ROOT detection (Going up until we find 'data' or reach /)
        current = Path(__file__).resolve().parent
        project_root = None
        for _ in range(10): # Max 10 levels
            if (current / "data").exists() and (current / "ml-service").exists():
                project_root = current
                break
            current = current.parent
            if current == current.parent: break


        if project_root:
            self.processed_dir = project_root / "data" / "processed"
        else:
            self.processed_dir = Path("data/processed")


    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self):
        """Execute the full training sequence."""
        logger.info("Starting fresh training pipeline...")

        # 1. Load Data (JSON-First)
        logger.info("Ingesting data from %s", self.processed_dir)
        X, y = DATASET_LOADER.load_from_json_dir(self.processed_dir, self.config)
        
        logger.info("Dataset Ready: %d samples, %d features", X.shape[0], X.shape[1])
        logger.info("Label Distribution:\n%s", y.value_counts())

        # 2. Get Model from Registry
        arch = self.config.get("active_model", "logistic_regression")
        model_id = self.config.get("deployment", {}).get("active_model_id", "MODEL_V1")
        
        logger.info("Initializing architecture: %s", arch)
        meta_cfg = self.config.get("deployment", {}).get("model_metadata", {})
        model_metadata = ModelMetadata(
            model_id=model_id,
            name=meta_cfg.get("name", "Unnamed Model"),
            version=meta_cfg.get("version", "1.0.0"),
            description=meta_cfg.get("description", "Freshly trained model")
        )
        
        model_wrapper = ModelRegistry.get_model(arch)
        model_wrapper.metadata = model_metadata

        # 3. Train
        logger.info("Training %s model...", arch)
        model_wrapper.train(X, y)

        # 4. Deploy Artifacts
        artifact_path = Path("ml_engine/ml/artifacts") / f"{model_id}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Deploying artifacts to %s", artifact_path)
        model_wrapper.save(artifact_path)

        logger.info("✅ Training sequence completed successfully.")


if __name__ == "__main__":
    _HERE = Path(__file__).resolve().parent
    _CONFIG = _HERE.parent / "configs" / "training_config.yaml"
    
    pipeline = TrainingPipeline(_CONFIG)
    pipeline.run()