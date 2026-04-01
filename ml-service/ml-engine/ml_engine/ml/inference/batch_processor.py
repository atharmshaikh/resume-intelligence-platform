# pyre-ignore-all-errors
"""
batch_processor.py
==================
Industry-grade Batch Inference Engine.
- Automates the full lifecycle: Uploads -> Processing -> Inference -> Cleanup.
- Enforces standardized storage structure.
- Integrated automated shredding (cleanup) of raw resumes.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ml_engine.ml.pipelines.parsing import ResumePipeline
from ml_engine.ml.inference.predictor import ResumePredictor
from ml_engine.utils.cleanup import CleanupService

logger = logging.getLogger(__name__)

def process_batch(config_path: str | None = None) -> None:
    """
    Standardized operational flow for the Resume Intelligence Platform.
    """
    # 1. Initialize Components
    try:
        predictor = ResumePredictor(config_path)
        pipeline = ResumePipeline()
        
        # Load Runtime Config
        runtime = predictor.config.get("runtime", {})
        storage = runtime.get("storage", {})
        cleanup_cfg = runtime.get("cleanup", {})
        
        # ── Standardized Central Storage Resolution ───────────────────
        # Here we prioritize absolute resolution for cross-platform stability.
        _HERE = Path(__file__).resolve().parent
        _ROOT = _HERE.parent.parent.parent.parent.parent # Project Root
        
        uploads_dir = (_ROOT / storage.get("uploads", "data/uploads")).resolve()
        processed_dir = (_ROOT / storage.get("processed", "data/processed")).resolve()
        results_dir = (_ROOT / storage.get("results", "data/results")).resolve()
        
        # Ensure directories exist (Standardizing the volume)
        for d in [uploads_dir, processed_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    except Exception as exc:
        logger.error(f"Failed to initialize Decoupled Batch Engine: {exc}")
        sys.exit(1)

    # 2. Iterate and Process Uploads
    resumes = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.docx"))
    
    if not resumes:
        logger.info("No new resumes found in %s", uploads_dir)
    else:
        logger.info("Found %d resumes to process.", len(resumes))

    for resume_path in resumes:
        try:
            # A. Parse to Optimized Data
            optimized_data = pipeline.parse(resume_path)
            
            # B. Save Optimized Data (Processed Layer)
            processed_path = processed_dir / f"{resume_path.stem}_features.json"
            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(optimized_data, f, indent=2, ensure_ascii=False)
            
            # C. Inference
            # The predictor expects a features dict (the 166 vector)
            inference_json = predictor.predict(optimized_data["features"])
            inference_obj = json.loads(inference_json)
            
            # D. Save Final Result (Results Layer)
            result_path = results_dir / f"{resume_path.stem}_decision.json"
            with open(result_path, "w", encoding="utf-8") as f:
                # Add identity context to the result for easier backend fetching
                inference_obj["identity"] = optimized_data["identity"]
                json.dump(inference_obj, f, indent=2, ensure_ascii=False)
                
            logger.info("Successfully processed and predicted: %s", resume_path.name)
            
        except Exception as exc:
            logger.error("Failed to process resume %s: %s", resume_path.name, exc)

    # 3. Automated Cleanup
    if cleanup_cfg.get("enabled", True):
        retention = cleanup_cfg.get("retention_hours", 24)
        cleaner = CleanupService(uploads_dir, retention)
        purged = cleaner.purge_stale_files()
        if purged > 0:
            logger.info("Automated cleanup completed. Purged %d stale resumes.", purged)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    process_batch(cfg)
