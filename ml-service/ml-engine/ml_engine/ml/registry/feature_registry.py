"""
Feature Registry

Purpose
-------
Central registry for ML features.

Benefits
--------
- Avoids hardcoded feature pipelines
- Enables scalable feature engineering (100+ features)
- Supports plug-and-play feature modules
- Ensures reproducibility of datasets

Design Principles
-----------------
- No dependency on parser or pipeline
- Safe execution
- Clear error messages
- Low overhead
"""

from typing import Callable, Dict, Any
import threading


class FeatureRegistryError(Exception):
    """Base exception for registry errors."""
    pass


class DuplicateFeatureError(FeatureRegistryError):
    """Raised when a feature is registered twice."""
    pass


class FeatureExecutionError(FeatureRegistryError):
    """Raised when a feature fails during computation."""
    pass


class FeatureRegistry:
    """
    Global feature registry.

    Stores feature computation functions.

    Example
    -------
    @FEATURE_REGISTRY.register("skills_count")
    def compute_skills_count(features):
        return features.get("skills_count", 0)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------
    # Register Feature
    # --------------------------------------------------

    def register(self, name: str) -> Callable:
        """
        Register a new feature.

        Parameters
        ----------
        name : str
            Unique feature name.

        Returns
        -------
        decorator
        """

        if not isinstance(name, str) or not name:
            raise ValueError("Feature name must be a non-empty string")

        def decorator(func: Callable):

            with self._lock:

                if name in self._registry:
                    raise DuplicateFeatureError(
                        f"Feature '{name}' already registered"
                    )

                self._registry[name] = func

            return func

        return decorator

    # --------------------------------------------------
    # Get Registered Features
    # --------------------------------------------------

    def get_registered_features(self) -> Dict[str, Callable]:
        """Return all registered features."""
        return dict(self._registry)

    # --------------------------------------------------
    # Compute Feature Vector
    # --------------------------------------------------

    def compute(self, feature_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all registered features.

        Parameters
        ----------
        feature_input : dict
            Input feature dictionary.

        Returns
        -------
        dict
            Computed feature vector.
        """

        if not isinstance(feature_input, dict):
            raise TypeError("Feature input must be a dictionary")

        result: Dict[str, Any] = {}

        for feature_name, func in self._registry.items():

            try:
                value = func(feature_input)

                if value is None:
                    value = 0

                result[feature_name] = value

            except Exception as exc:

                raise FeatureExecutionError(
                    f"Feature '{feature_name}' failed: {exc}"
                ) from exc

        return result

    # --------------------------------------------------
    # Feature Count
    # --------------------------------------------------

    def size(self) -> int:
        """Return number of registered features."""
        return len(self._registry)

    # --------------------------------------------------
    # Clear Registry (Testing)
    # --------------------------------------------------

    def clear(self) -> None:
        """Clear registry (for testing only)."""
        with self._lock:
            self._registry.clear()


# Global singleton registry
FEATURE_REGISTRY = FeatureRegistry()