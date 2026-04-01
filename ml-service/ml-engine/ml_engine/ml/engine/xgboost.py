# pyre-ignore-all-errors
"""
xgboost.py
==========
Experimental XGBoost model shell for the Resume Pipeline.
"""

import logging
from typing import Optional
from pathlib import Path

from ..core.base_model import BaseModel
from ..core.model_metadata import ModelMetadata

logger = logging.getLogger(__name__)

class XGBoostError(Exception):
    """Raised on XGBoost model errors."""
    pass

class XGBoostModel(BaseModel):
    """
    Experimental shell for XGBoost integration.
    Placeholder for future "PROD" capability.
    """
    
    def __init__(self, metadata: Optional[ModelMetadata] = None, **kwargs) -> None:
        super().__init__(metadata)
        self.model = None
        self._trained = False

    def train(self, X, y):
        raise NotImplementedError(
            "XGBoost integration is experimental. "
            "Please install 'xgboost' to activate training."
        )

    def predict(self, X):
        raise NotImplementedError("XGBoost integration is experimental.")

    def save(self, path: str | Path):
        logger.info("XGBoost metadata saved to placeholder.")
        pass

    def load(self, path: str | Path):
        logger.warning("XGBoost is currently an experimental shell and not fully loaded.")
        self._trained = False
