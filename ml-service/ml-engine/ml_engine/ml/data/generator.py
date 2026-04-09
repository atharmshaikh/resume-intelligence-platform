# pyre-ignore-all-errors
"""
generator.py
============
Synthetic dataset generation for training the Resume ML Engine.
Replaced the old synthetic_dataset.py.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Dict, List

from .builder import DATASET_BUILDER

logger = logging.getLogger(__name__)


class GeneratorError(Exception):
    """Raised on dataset generation failure."""
    pass


class SyntheticDatasetGenerator:
    """
    Generates large-scale synthetic resume feature datasets.
    Uses DATASET_BUILDER to ensure features are balanced and ML-ready.
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        _HERE = Path(__file__).resolve().parent
        # Go up to ml-service/ml-engine/ and then down to datasets/
        _ML_ROOT = _HERE.parent.parent.parent
        self.output_path = Path(output_path or (_ML_ROOT / "ml_engine" / "ml" / "datasets" / "resume_dataset.csv"))
        self.builder = DATASET_BUILDER

    def generate(self, n_samples: int = 1000) -> None:
        """Generate and save n_samples to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict] = []
        
        # ── Weighted generation ───────────────────────────────────
        # We want roughly 33/33/33 balanced classes for training
        classes = [0, 1, 2]
        per_class = n_samples // 3

        for label in classes:
            for _ in range(per_class):
                raw = self._generate_raw_features(label)
                row = self.builder.build_row(raw, label)
                rows.append(row)

        # ── Write CSV ─────────────────────────────────────────────
        if not rows:
            raise GeneratorError("No data generated")

        fieldnames = list(rows[0].keys())
        try:
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Successfully generated %d samples to %s", len(rows), self.output_path)
        except Exception as exc:
            raise GeneratorError(f"Failed to write CSV: {exc}")

    def _generate_raw_features(self, label: int) -> Dict:
        """
        Produce a randomized feature dict biased towards the target label.
        - 0 (Weak)    : low scores, many penalties
        - 1 (Average) : moderate scores
        - 2 (Strong)  : high scores, top institues, multiple projects
        """
        feat: Dict = {}

        # G01 Contact (Base)
        feat["has_name"] = 1
        feat["has_email"] = 1 if label > 0 else random.choice([0, 1])
        feat["has_phone"] = 1 if label > 0 else random.choice([0, 1])
        feat["has_location"] = random.choice([0, 1])
        feat["contact_completeness_score"] = random.uniform(0.5, 1.0) if label > 0 else random.uniform(0.1, 0.6)

        # G04 Education
        if label == 2:
            feat["has_bachelor_degree"] = 1
            feat["is_cs_it_candidate"] = 1
            feat["has_top_institution"] = random.choice([0, 1])
            feat["cgpa_value"] = random.uniform(8.0, 10.0)
        elif label == 1:
            feat["has_bachelor_degree"] = 1
            feat["is_cs_it_candidate"] = random.choice([0, 1])
            feat["cgpa_value"] = random.uniform(6.5, 8.5)
        else:
            feat["has_bachelor_degree"] = random.choice([0, 1])
            feat["cgpa_value"] = random.uniform(5.0, 7.0)

        # G03 Skills
        if label == 2:
            feat["skills_count"] = random.randint(8, 15)
            feat["skill_category_count"] = random.randint(3, 6)
            feat["has_ai_ml_skills"] = 1
        elif label == 1:
            feat["skills_count"] = random.randint(4, 9)
            feat["skill_category_count"] = random.randint(2, 4)
        else:
            feat["skills_count"] = random.randint(1, 5)
            feat["skill_category_count"] = random.randint(0, 2)

        # G05 Experience
        if label == 2:
            feat["experience_years_estimate"] = random.uniform(2.0, 5.0)
        elif label == 1:
            feat["experience_years_estimate"] = random.uniform(0.5, 2.5)
        else:
            feat["experience_years_estimate"] = random.uniform(0, 1.0)
            feat["is_fresher"] = 1

        # G09 ATS Quality
        feat["resume_word_count"] = random.randint(300, 600) if label > 0 else random.randint(50, 200)

        # G10 Penalties (Inverse)
        if label == 0:
            feat["ats_penalty_no_skills"] = random.choice([0, 1])
            feat["ats_penalty_unprofessional_email"] = random.choice([0, 1])

        # Composite
        if label == 2:
            feat["candidate_readiness_score"] = random.uniform(85, 100)
        elif label == 1:
            feat["candidate_readiness_score"] = random.uniform(65, 84)
        else:
            feat["candidate_readiness_score"] = random.uniform(10, 64)

        return feat