from abc import ABC, abstractmethod
from typing import Optional
from .model_metadata import ModelMetadata

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the Resume Platform.
    Enforces training, prediction, and serialization contracts.
    """
    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        self.metadata = metadata

    @abstractmethod
    def train(self, X, y):
        """Train the model on features X and labels y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict labels for input X."""
        pass

    @abstractmethod
    def save(self, path):
        """Serialize model and metadata to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model and metadata from disk."""
        pass
