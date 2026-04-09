"""
tests/test_pipeline.py
======================
Integration test + explained output for ResumePipeline.
"""

from __future__ import annotations

import json
import sys
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Ensure ml_engine is importable when run directly (not via pytest/install)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ML_ENGINE_ROOT = _HERE.parent
if str(_ML_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ENGINE_ROOT))

# Import the core pipeline
from ml_engine.pipeline import ResumePipeline # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_resumes(base_dir: Path) -> List[Path]:
    """Search for PDF/Docx resumes in the data/uploads directory."""
    # Standardized uploads are stored at repo-root/data/uploads
    repo_root = base_dir.parent.parent
    resume_dir = repo_root / "data" / "uploads"
    
    if not resume_dir.exists():
        msg = f"Standardized uploads directory not found at: {resume_dir}. Please add resumes to begin testing."
        print(f"⚠️  {msg}")
        return []

    resumes = list(resume_dir.glob("*.pdf")) + list(resume_dir.glob("*.docx"))
    return [p for p in resumes if p.exists() and p.is_file() and p.stat().st_size > 0]


def _explain_output(result_dict: dict) -> None:
    """Print a human-readable, section-by-section explanation of the result."""

    sep = "─" * 60
    identity = result_dict.get("identity", {}) if isinstance(result_dict.get("identity", {}), dict) else {}
    quality = result_dict.get("quality_metrics", result_dict.get("quality", {}))
    features = result_dict.get("features", {})

    # ── 1. Candidate identity ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  📋  CANDIDATE IDENTITY")
    print(sep)
    print(f"  Name     : {identity.get('name', '—')}")
    print(f"  Email    : {identity.get('email', '—')}")
    print(f"  Phone    : {identity.get('phone', '—')}")
    print(f"  Location : {identity.get('location', '—')}")

    # ── 2. Resume quality scores ──────────────────────────────────────────
    scores = result_dict.get("scores", {})
    if not scores: # Fallback for v1.1 reporting
        scores = {
            "ats_score": features.get("overall_profile_strength", 0),
            "typo_score": 100 - (features.get("typo_count", 0) * 5)
        }

    print(f"\n{sep}")
    print("  📊  QUALITY SCORES  (0–100, higher = better)")
    print(sep)
    print(f"  Overall Score    : {scores.get('ats_score', 0):.1f} / 100")
    print(f"  Typo Score       : {scores.get('typo_score', 0):.1f} / 100  "
          f"({features.get('typo_count', 0)} typos detected)")

    # ── 3. ML Features summary ────────────────────────────────────────────
    if features:
        print(f"\n{sep}")
        print("  🤖  ML FEATURES SUMMARY (165 features for Zero-Trust CI/CD)")
        print(sep)

        # Composite scores
        print("\n  [ Composite Scores ]")
        print(f"    Candidate Readiness Score  : {features.get('candidate_readiness_score', 0):.1f} / 100")
        print(f"    Overall Profile Strength   : {features.get('overall_profile_strength', 0):.1f} / 100")
        print(f"    Skills Sub-score           : {features.get('skills_subscore', 0):.1f} / 100")

        # Skills breakdown
        print("\n  [ Skills (G03) ]")
        print(f"    Total skill keywords       : {features.get('skills_count', 0)}")
        print(f"    Skill weight score         : {features.get('skill_weight_score', 0):.2f}")

        # Experience
        print("\n  [ Experience (G05) ]")
        print(f"    Has experience             : {'✅ Yes' if features.get('has_experience') else '❌ No'}")
        print(f"    Has internship             : {'✅ Yes' if features.get('has_internship') else '❌ No'}")
        print(f"    Estimated years            : {features.get('years_of_experience', 0):.1f} yrs")

        # ATS Penalties
        total_penalty = features.get("ats_total_penalty_score", 0)
        print(f"\n  [ ATS Penalty Flags (G10) ] — Total: {total_penalty}")
        if total_penalty == 0:
            print("    ✅ No penalty flags — clean resume!")
        else:
             print(f"    ⚠️  Detected {total_penalty} penalty point(s)")

    # ── 4. Overall verdict ────────────────────────────────────────────────
    ml_pred = result_dict.get("ml_prediction", {})
    if isinstance(ml_pred, str):
        try:
            ml_pred = json.loads(ml_pred)
        except:
             ml_pred = {}

    print(f"\n{sep}")
    print("  🏆  OVERALL VERDICT")
    print(sep)
    decision = ml_pred.get("decision", "Unknown")
    confidence = ml_pred.get("model_output", {}).get("confidence", 0)
    
    print(f"    Decision  : {decision}")
    print(f"    Confidence: {confidence:.2f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main runner (direct python execution)
# ---------------------------------------------------------------------------

def _run_pipeline() -> None:
    pipeline = ResumePipeline()
    base_dir = Path(__file__).resolve().parents[1]
    resume_files = _find_resumes(base_dir)

    for resume_file in resume_files:
        print(f"\n{'═'*60}")
        print(f"  📄  Processing: {resume_file.name}")
        print(f"{'═'*60}")

        result_dict = pipeline.process(resume_file)

        if result_dict is None:
            print("  ⚠️  Skipped: failed validation or hard rules")
            continue

        _explain_output(result_dict)

        print("  [ Full JSON output ]")
        # Truncate some large lists for cleaner console output
        clean_result = dict(result_dict)
        if "normalized_resume" in clean_result:
             nr = clean_result["normalized_resume"]
             for k in ["skills", "experience", "projects"]:
                  if k in nr and len(nr[k]) > 5:
                       nr[k] = nr[k][:5] + ["..."]
        
        print(json.dumps(clean_result, indent=2, ensure_ascii=False))


def main() -> None:
    try:
        _run_pipeline()
        print("\n✅  Test finished successfully\n")
    except Exception as exc:
        print(f"\n❌  Test failed: {exc}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# pytest-compatible test function
# ---------------------------------------------------------------------------

def test_pipeline_runs_successfully() -> None:
    """
    Pytest test: verifies that the pipeline processes all sample resumes
    without errors and returns a valid feature dictionary.
    """
    pipeline = ResumePipeline()
    base_dir = Path(__file__).resolve().parents[1]
    resume_files = _find_resumes(base_dir)

    if not resume_files:
         pytest.skip("No sample resumes found for testing.")

    for resume_file in resume_files:
        result = pipeline.process(resume_file)
        
        # Basic structural assertions
        assert isinstance(result, dict), "Result must be a dict"
        assert "identity" in result, "Result must have 'identity' context"
        assert "features" in result, "Result must have 'features' for ML"

        # Check for 100% type safety and structure
        feats = result["features"]
        assert isinstance(feats, dict), "features must be a dict"
        assert len(feats) >= 150, f"Expected 150+ features, got {len(feats)}"

        # Score sanity
        readiness = feats.get("candidate_readiness_score", -1)
        assert 0 <= readiness <= 100, f"Readiness score out of range: {readiness}"
        
        print(f"  ✅  {resume_file.name}  →  readiness={readiness:.1f}")


if __name__ == "__main__":
    main()
