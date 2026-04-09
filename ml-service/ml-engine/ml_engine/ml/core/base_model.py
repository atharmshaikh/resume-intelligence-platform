from abc import ABC, abstractmethod
from typing import Optional, Any
from .model_metadata import ModelMetadata

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the Resume Platform.
    Enforces training, prediction, and serialization contracts.
    """
    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        self.metadata = metadata

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model on features X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict labels for input X."""
        pass

    @abstractmethod
    def save(self, path: Any) -> None:
        """Serialize model and metadata to disk."""
        pass

    @abstractmethod
    def load(self, path: Any) -> None:
        """Load model and metadata from disk."""
        pass
