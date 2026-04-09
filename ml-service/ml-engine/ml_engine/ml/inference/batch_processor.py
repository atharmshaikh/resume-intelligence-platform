# pyre-ignore-all-errors
"""
batch_processor.py
==================
ML-ready Feature Dataset Generator.

Produces clean, deterministic feature datasets for ML training.
"""

import sys
import json
import logging
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ml_engine.ml.pipelines.parsing import ResumePipeline
from ml_engine.ml.inference.predictor import ResumePredictor

logger = logging.getLogger(__name__)


def generate_dataset(resumes: List[dict]) -> List[Dict[str, Any]]:
    """
    Generate ML-ready dataset from parsed resume data.

    Args:
        resumes: List of parsed resume dictionaries with 'features' key

    Returns:
        List of flat feature dictionaries (ML-ready)
    """
    dataset = []

    for resume in resumes:
        features = resume.get("features")
        if not features:
            continue

        label = resume.get("label")

        row = dict(features)

        if label is not None:
            row["label"] = label

        dataset.append(row)

    return dataset


def _resolve_runtime_paths(config_path: str | None) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    runtime_cfg: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            runtime_cfg = yaml.safe_load(f) or {}
    runtime = runtime_cfg.get("runtime", {})
    storage = runtime.get("storage", {})

    _here = Path(__file__).resolve().parent
    # ml_engine/ml/inference -> ml-service/ml-engine
    _root = _here.parent.parent.parent  
    uploads_dir = (_root.parent.parent / storage.get("uploads", "data/uploads")).resolve()
    processed_dir = (_root.parent.parent / storage.get("processed", "data/processed")).resolve()
    results_dir = (_root.parent.parent / storage.get("results", "data/results")).resolve()
    return uploads_dir, processed_dir, results_dir


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def process_batch(config_path: str | None = None) -> None:
    """
    Standardized operational flow for the Resume Intelligence Platform.
    - Parser output → processed/
    - ML prediction output → results/
    """
    try:
        pipeline = ResumePipeline()
        predictor = ResumePredictor(config_path)
        uploads_dir, processed_dir, results_dir = _resolve_runtime_paths(config_path)

        for d in [uploads_dir, processed_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    except Exception as exc:
        logger.error(f"Failed to initialize Decoupled Batch Engine: {exc}")
        sys.exit(1)

    all_resumes = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.docx"))
    resumes = [p for p in all_resumes if p.exists() and p.is_file() and p.stat().st_size > 0]

    if len(resumes) < len(all_resumes):
        skipped = len(all_resumes) - len(resumes)
        logger.info("Skipping %d empty resume file(s) from uploads.", skipped)

    if not resumes:
        logger.info("No new resumes found in %s", uploads_dir)
    else:
        logger.info("Found %d resumes. Starting batch processing...", len(resumes))

        for resume_path in resumes:
            try:
                slug_name = re.sub(r"[^a-z0-9]+", "-", resume_path.stem.lower()).strip("-")

                # ── Parser Output → processed/ ────────────────────────
                result = pipeline.parse(resume_path)

                if result is None:
                    logger.info("Skipping %s: failed validation or hard rules", slug_name)
                    continue

                features = result.get("features", {})

                # Save FULL parser output to processed/
                parser_output = {
                    "identity": result.get("identity", {}),
                    "normalized_resume": result.get("normalized_resume", {}),
                    "features": features
                }
                feature_file = processed_dir / f"{slug_name}-features.json"
                _write_json(feature_file, parser_output)

                # ── ML Prediction Output → results/ ────────────────────
                ml_result_json = predictor.predict(features)
                ml_result = json.loads(ml_result_json)
                
                result_output = {
                    "candidate": result.get("identity", {}),
                    "ml_prediction": ml_result
                }
                result_file = results_dir / f"{slug_name}-result.json"
                _write_json(result_file, result_output)

                logger.info("✅ Processed: %s → %s", slug_name, ml_result.get("decision", "Unknown"))

            except Exception as exc:
                if "empty" in str(exc).lower():
                    logger.warning("Skipping empty resume file: %s", resume_path.name)
                    continue
                logger.error("Failed to process resume %s: %s", resume_path.name, exc)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    process_batch(cfg)
