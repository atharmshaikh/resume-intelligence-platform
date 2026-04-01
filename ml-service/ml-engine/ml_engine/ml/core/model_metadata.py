from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    framework: str = "scikit-learn"
    features_count: int = 166
    changelog: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Save metadata to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelMetadata":
        """Load metadata from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(**data)
