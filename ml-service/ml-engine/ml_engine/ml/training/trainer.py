"""
Trainer

Handles model training and artifact saving.
"""

from pathlib import Path

from ..dataset import DATASET_LOADER
from ..models import ModelRegistry


class TrainerError(Exception):
    pass


class Trainer:

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.model = ModelRegistry.get_model(model_name)

    # --------------------------------------------------

    def train(self, dataset_path: str):

        X, y = DATASET_LOADER.load(dataset_path)

        if len(X) == 0:
            raise TrainerError("Dataset is empty")

        self.model.train(X, y)

        return self.model

    # --------------------------------------------------

    def save_model(self, model, artifact_path: Path):

        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(artifact_path)