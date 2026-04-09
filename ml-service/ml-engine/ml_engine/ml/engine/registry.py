"""
registry.py
===========
Central registry for ML model architectures.
Supports dynamic instantiation with hyperparameter injection.
"""

from typing import Dict, Type
from ..core.base_model import BaseModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel
from .logistic_regression import LogisticRegressionModel

class ModelRegistry:
    """
    Registry of available model architectures.
    Mapping between logical names and implementation classes.
    """

    MODELS: Dict[str, Type[BaseModel]] = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "logistic_regression": LogisticRegressionModel
    }

    @classmethod
    def get_model(cls, name: str, **kwargs) -> BaseModel:
        """
        Instantiate a model class by name.
        Allows passing arbitrary hyperparameters via kwargs.
        """
        if name not in cls.MODELS:
            raise ValueError(
                f"Unknown model architecture: '{name}'. "
                f"Available: {list(cls.MODELS.keys())}"
            )

        return cls.MODELS[name](**kwargs)