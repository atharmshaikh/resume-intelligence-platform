# pyre-ignore-all-errors
"""
loader.py
=========
JSON-First Dataset loading and processing for training.
Eliminates CSV dependency by reading directly from data/processed/.
"""

from __future__ import annotations

import json
import logging
import glob
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd  # type: ignore  # type: ignore  # type: ignore
from ..schemas.feature_schema import FEATURE_LIST

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised on dataset loading failure."""
    pass


class DatasetLoader:
    """
    Handles JSON dataset ingestion from data/processed/.
    Calculates labels on-the-fly based on YAML labeling rules.
    """

    def load_from_json_dir(self, processed_dir: str | Path, config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Scan directory for JSONs and return (X, y) for training.
        """
        processed_dir = Path(processed_dir)
        if not processed_dir.exists():
             raise LoaderError(f"Processed directory not found: {processed_dir}")

        json_paths = glob.glob(str(processed_dir / "*.json"))
        if not json_paths:
             raise LoaderError(f"No processed JSONs found in {processed_dir}")

        logger.info("Found %d processed resumes for training", len(json_paths))

        rows = []
        labels = []
        
        rules = config.get("labeling_rules", {})

        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                features = data.get("features", {})
                if not features:
                    continue

                # Ensure all features in FEATURE_LIST are present
                row = {f: features.get(f, 0) for f in FEATURE_LIST}
                
                # Dynamic Labeling based on YAML RULES
                label = self._calculate_label(features, rules)
                
                rows.append(row)
                labels.append(label)
                
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        if not rows:
            raise LoaderError("No valid training samples extracted from JSONs.")

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name="label")

        return X, y

    def _calculate_label(self, features: Dict, rules: Dict) -> int:
        """
        Applies strict YAML-driven heuristics to determine the ground-truth label.
        """
        strong = rules.get("strong_candidate", {})
        avg = rules.get("average_candidate", {})

        score = float(features.get("score", 0))
        projects = int(features.get("projects_count", 0))
        exp = float(features.get("years_of_experience", 0))
        has_intern = int(features.get("has_internship", 0))
        skills_count = int(features.get("skills_count", 0))
        typos = int(features.get("typo_count", 0))

        # 1. Strong Candidate Check
        is_strong = (
            score >= strong.get("min_score", 85) and
            (projects >= strong.get("min_projects", 2) or 
             (exp >= strong.get("min_experience_years", 1.0) or (strong.get("allow_internship_as_experience") and has_intern)))
        )
        
        # Stricter penalty if configured
        if strong.get("require_zero_typos") and typos > 0:
            is_strong = False

        if is_strong:
            return 2
            
        # 2. Average Candidate Check
        is_avg = (
            score >= avg.get("min_score", 65) or
            has_intern or
            skills_count >= avg.get("min_skills", 15) or
            avg.get("allow_fresher", False)
        )
        
        if is_avg:
            return 1

        # 3. Weak Candidate
        return 0


DATASET_LOADER = DatasetLoader()
