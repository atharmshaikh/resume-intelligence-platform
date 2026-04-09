"""
tests/test_pipeline.py
======================
Integration test + explained output for ResumePipeline.

Run methods
-----------
  # Easy method (one command from project root):
  bash ml-service/ml-engine/run_test.sh

  # Direct python (from ml-engine/ directory):
  ../../.venv/bin/python tests/test_pipeline.py

  # pytest (from ml-engine/ directory):
  ../../.venv/bin/pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
<<<<<<< HEAD
=======
from typing import List
>>>>>>> feature/optimization-and-refactor

# ---------------------------------------------------------------------------
# Ensure ml_engine is importable when run directly (not via pytest/install)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ML_ENGINE_ROOT = _HERE.parent
if str(_ML_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ENGINE_ROOT))

<<<<<<< HEAD
from ml_engine.pipeline import ResumePipeline  # noqa: E402  (after sys.path fix)
=======
from ml_engine.ml.pipelines.parsing import ResumePipeline  # noqa: E402  (after sys.path fix)
>>>>>>> feature/optimization-and-refactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
<<<<<<< HEAD

def _find_resumes(base_dir: Path) -> list[Path]:
    resume_dir = base_dir / "sample_resumes"
    if not resume_dir.exists():
        raise FileNotFoundError(f"sample_resumes/ not found at: {base_dir}")
    files = sorted(
        f for f in resume_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".pdf", ".docx"}
    )
    if not files:
        raise FileNotFoundError("No .pdf / .docx files found in sample_resumes/")
    return files


def _explain_output(result_dict: dict) -> None:
    """Print a human-readable, section-by-section explanation of the result."""

    sep = "─" * 60

    # ── 1. Candidate identity ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  📋  CANDIDATE IDENTITY")
    print(sep)
    print(f"  Name     : {result_dict.get('name', '—')}")
    print(f"  Email    : {result_dict.get('email', '—')}")
    print(f"  Phone    : {result_dict.get('phone', '—')}")
    print(f"  Location : {result_dict.get('location', '—')}")

    # ── 2. Resume quality scores ──────────────────────────────────────────
    scores = result_dict.get("scores", {})
    print(f"\n{sep}")
    print("  📊  QUALITY SCORES  (0–100, higher = better)")
    print(sep)
    print(f"  ATS Score        : {scores.get('ats_score', 0):.1f} / 100")
    print(f"  Contact Score    : {scores.get('contact_score', 0):.1f} / 100")
    print(f"  Structure Score  : {scores.get('structure_score', 0):.1f} / 100")
    print(f"  Content Score    : {scores.get('content_score', 0):.1f} / 100")
    print(f"  Length Score     : {scores.get('length_score', 0):.1f} / 100")
    print(f"  Typo Score       : {scores.get('typo_score', 0):.1f} / 100  "
          f"({result_dict.get('quality', {}).get('typos', {}).get('typo_count', 0)} typos detected)")

    # ── 3. ML Features summary ────────────────────────────────────────────
    feats = result_dict.get("features", {})
    if feats:
        print(f"\n{sep}")
        print("  🤖  ML FEATURES SUMMARY (165 features for Random Forest / XGBoost)")
        print(sep)

        # Composite scores
        print("\n  [ Composite Scores ]")
        print(f"    Candidate Readiness Score  : {feats.get('candidate_readiness_score', 0):.1f} / 100"
              f"  (raw {feats.get('raw_positive_score', 0):.1f} − penalty {feats.get('penalty_deduction', 0):.1f})")
        print(f"    Overall Profile Strength   : {feats.get('overall_profile_strength', 0):.1f} / 100")
        print(f"    Skills Sub-score           : {feats.get('skills_subscore', 0):.1f} / 100")
        print(f"    Education Sub-score        : {feats.get('education_subscore', 0):.1f} / 100")
        print(f"    Experience Sub-score       : {feats.get('experience_subscore', 0):.1f} / 100")
        print(f"    Projects Sub-score         : {feats.get('projects_subscore', 0):.1f} / 100")

        # Skills breakdown
        print("\n  [ Skills (G03) ]")
        print(f"    Total skill keywords       : {feats.get('total_skills', 0)}")
        print(f"    Skill categories covered   : {feats.get('skill_category_count', 0)} / 8")
        print(f"    Skill weight score         : {feats.get('skill_weight_score', 0):.2f}")
        print(f"    Has high-value skills      : {'✅ Yes' if feats.get('has_high_value_skills') else '❌ No'}")
        print(f"    Cloud / DevOps             : {'✅' if feats.get('has_cloud_skills') else '—'}  "
              f"  AI/ML : {'✅' if feats.get('has_ml_skills') else '—'}  "
              f"  Security : {'✅' if feats.get('has_security_skills') else '—'}")

        # Education
        print("\n  [ Education (G04) ]")
        print(f"    Has Bachelor degree        : {'✅ Yes' if feats.get('has_bachelor_degree') else '❌ No'}")
        print(f"    Has Master degree          : {'✅ Yes' if feats.get('has_master_degree') else '❌ No'}")
        print(f"    CGPA detected              : {'✅' if feats.get('has_cgpa') else '❌'}  "
              f"Value: {feats.get('cgpa_value', 0):.2f}")
        print(f"    Strong academics (≥8 CGPA) : {'✅ Yes' if feats.get('has_strong_academics') else '❌ No'}")
        print(f"    Top institution            : {'✅ Yes' if feats.get('has_top_institution') else '❌ No'}")

        # Experience
        print("\n  [ Experience (G05) ]")
        print(f"    Has experience             : {'✅ Yes' if feats.get('has_experience') else '❌ No'}")
        print(f"    Has internship             : {'✅ Yes' if feats.get('experience_has_internship') else '❌ No'}")
        print(f"    Estimated years            : {feats.get('experience_years_estimate', 0):.1f} yrs")
        print(f"    Quantified impact          : {'✅ Yes' if feats.get('has_quantified_impact_in_exp') else '❌ No'}")

        # Projects
        print("\n  [ Projects (G06) ]")
        print(f"    Project count              : {feats.get('projects_count', 0)}")
        print(f"    Tech stack diversity       : {feats.get('project_tech_stack_count', 0)}")
        print(f"    Has deployed project       : {'✅ Yes' if feats.get('has_deployed_project') else '❌ No'}")

        # Online presence
        print("\n  [ Online Presence (G08) ]")
        print(f"    LinkedIn                   : {'✅' if feats.get('has_linkedin') else '❌'}")
        print(f"    GitHub                     : {'✅' if feats.get('has_github') else '❌'}")
        print(f"    Portfolio                  : {'✅' if feats.get('has_portfolio') else '❌'}")

        # ATS Penalties
        total_penalty = feats.get("ats_total_penalty_score", 0)
        penalty_flags = [k for k, v in feats.items() if k.startswith("ats_penalty_") and v == 1]
        print(f"\n  [ ATS Penalty Flags (G10) ] — Total: {total_penalty}")
        print(f"    Penalty deduction          : -{feats.get('penalty_deduction', 0):.1f} pts from raw score")
        if penalty_flags:
            for flag in penalty_flags:
                label = flag.replace("ats_penalty_", "").replace("_", " ").title()
                print(f"    ⚠️  {label}")
        else:
            print("    ✅ No penalty flags — clean resume!")

    # ── 4. Sections detected ──────────────────────────────────────────────
    sections = result_dict.get("sections", {})
    print(f"\n{sep}")
    print(f"  📂  SECTIONS DETECTED ({len(sections)} found)")
    print(sep)
    for sec_name, lines in sections.items():
        print(f"    [{sec_name.upper():20s}]  {len(lines)} line(s)")

    # ── 5. Overall verdict ────────────────────────────────────────────────
    readiness = feats.get("candidate_readiness_score", 0) if feats else 0
    print(f"\n{sep}")
    print("  🏆  OVERALL VERDICT")
    print(sep)
    if readiness >= 75:
        verdict = "🟢  Strong candidate — good ATS compatibility"
    elif readiness >= 55:
        verdict = "🟡  Average candidate — some improvements needed"
    else:
        verdict = "🔴  Weak candidate — significant improvements needed"
    print(f"    {verdict}")
    print(f"    Readiness Score: {readiness:.1f} / 100")
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

        result = pipeline.parse(resume_file)
        result_dict = result.to_dict()

        # Explained human-readable output
        _explain_output(result_dict)

        # Full JSON dump (for debugging)
=======

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

    # ── 1. Candidate identity ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  📋  CANDIDATE IDENTITY")
    print(sep)
    print(f"  Name     : {identity.get('name', result_dict.get('name', '—'))}")
    print(f"  Email    : {identity.get('email', result_dict.get('email', '—'))}")
    print(f"  Phone    : {identity.get('phone', result_dict.get('phone', '—'))}")
    print(f"  Location : {identity.get('location', result_dict.get('location', '—'))}")

    # ── 2. Resume quality scores ──────────────────────────────────────────
    scores = result_dict.get("scores", {})
    print(f"\n{sep}")
    print("  📊  QUALITY SCORES  (0–100, higher = better)")
    print(sep)
    print(f"  ATS Score        : {scores.get('ats_score', 0):.1f} / 100")
    print(f"  Contact Score    : {scores.get('contact_score', 0):.1f} / 100")
    print(f"  Structure Score  : {scores.get('structure_score', 0):.1f} / 100")
    print(f"  Content Score    : {scores.get('content_score', 0):.1f} / 100")
    print(f"  Length Score     : {scores.get('length_score', 0):.1f} / 100")
    print(f"  Typo Score       : {scores.get('typo_score', 0):.1f} / 100  "
          f"({quality.get('typos', {}).get('typo_count', 0)} typos detected)")

    # ── 3. ML Features summary ────────────────────────────────────────────
    feats = result_dict.get("features", {})
    if feats:
        print(f"\n{sep}")
        print("  🤖  ML FEATURES SUMMARY (165 features for Random Forest / XGBoost)")
        print(sep)

        # Composite scores
        print("\n  [ Composite Scores ]")
        print(f"    Candidate Readiness Score  : {feats.get('candidate_readiness_score', 0):.1f} / 100"
              f"  (raw {feats.get('raw_positive_score', 0):.1f} − penalty {feats.get('penalty_deduction', 0):.1f})")
        print(f"    Overall Profile Strength   : {feats.get('overall_profile_strength', 0):.1f} / 100")
        print(f"    Skills Sub-score           : {feats.get('skills_subscore', 0):.1f} / 100")
        print(f"    Education Sub-score        : {feats.get('education_subscore', 0):.1f} / 100")
        print(f"    Experience Sub-score       : {feats.get('experience_subscore', 0):.1f} / 100")
        print(f"    Projects Sub-score         : {feats.get('projects_subscore', 0):.1f} / 100")

        # Skills breakdown
        print("\n  [ Skills (G03) ]")
        print(f"    Total skill keywords       : {feats.get('total_skills', 0)}")
        print(f"    Skill categories covered   : {feats.get('skill_category_count', 0)} / 8")
        print(f"    Skill weight score         : {feats.get('skill_weight_score', 0):.2f}")
        print(f"    Has high-value skills      : {'✅ Yes' if feats.get('has_high_value_skills') else '❌ No'}")
        print(f"    Cloud / DevOps             : {'✅' if feats.get('has_cloud_skills') else '—'}  "
              f"  AI/ML : {'✅' if feats.get('has_ml_skills') else '—'}  "
              f"  Security : {'✅' if feats.get('has_security_skills') else '—'}")

        # Education
        print("\n  [ Education (G04) ]")
        print(f"    Has Bachelor degree        : {'✅ Yes' if feats.get('has_bachelor_degree') else '❌ No'}")
        print(f"    Has Master degree          : {'✅ Yes' if feats.get('has_master_degree') else '❌ No'}")
        print(f"    CGPA detected              : {'✅' if feats.get('has_cgpa') else '❌'}  "
              f"Value: {feats.get('cgpa_value', 0):.2f}")
        print(f"    Strong academics (≥8 CGPA) : {'✅ Yes' if feats.get('has_strong_academics') else '❌ No'}")
        print(f"    Top institution            : {'✅ Yes' if feats.get('has_top_institution') else '❌ No'}")

        # Experience
        print("\n  [ Experience (G05) ]")
        print(f"    Has experience             : {'✅ Yes' if feats.get('has_experience') else '❌ No'}")
        print(f"    Has internship             : {'✅ Yes' if feats.get('experience_has_internship') else '❌ No'}")
        print(f"    Estimated years            : {feats.get('experience_years_estimate', 0):.1f} yrs")
        print(f"    Quantified impact          : {'✅ Yes' if feats.get('has_quantified_impact_in_exp') else '❌ No'}")

        # Projects
        print("\n  [ Projects (G06) ]")
        print(f"    Project count              : {feats.get('projects_count', 0)}")

        # Online presence
        print("\n  [ Online Presence (G08) ]")
        print(f"    LinkedIn                   : {'✅' if feats.get('has_linkedin') else '❌'}")
        print(f"    GitHub                     : {'✅' if feats.get('has_github') else '❌'}")
        print(f"    Portfolio                  : {'✅' if feats.get('has_portfolio') else '❌'}")

        # ATS Penalties
        total_penalty = feats.get("ats_total_penalty_score", 0)
        penalty_flags = [k for k, v in feats.items() if k.startswith("ats_penalty_") and v == 1]
        print(f"\n  [ ATS Penalty Flags (G10) ] — Total: {total_penalty}")
        print(f"    Penalty deduction          : -{feats.get('penalty_deduction', 0):.1f} pts from raw score")
        if penalty_flags:
            for flag in penalty_flags:
                label = flag.replace("ats_penalty_", "").replace("_", " ").title()
                print(f"    ⚠️  {label}")
        else:
            print("    ✅ No penalty flags — clean resume!")

    # ── 4. Sections detected ──────────────────────────────────────────────
    sections = result_dict.get("sections", {})
    print(f"\n{sep}")
    print(f"  📂  SECTIONS DETECTED ({len(sections)} found)")
    print(sep)
    for sec_name, lines in sections.items():
        print(f"    [{sec_name.upper():20s}]  {len(lines)} line(s)")

    # ── 5. Overall verdict ────────────────────────────────────────────────
    readiness = feats.get("candidate_readiness_score", 0) if feats else 0
    print(f"\n{sep}")
    print("  🏆  OVERALL VERDICT")
    print(sep)
    if readiness >= 75:
        verdict = "🟢  Strong candidate — good ATS compatibility"
    elif readiness >= 55:
        verdict = "🟡  Average candidate — some improvements needed"
    else:
        verdict = "🔴  Weak candidate — significant improvements needed"
    print(f"    {verdict}")
    print(f"    Readiness Score: {readiness:.1f} / 100")
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

        result_dict = pipeline.parse(resume_file)

        if result_dict is None:
            print("  ⚠️  Skipped: failed validation or hard rules")
            continue

        _explain_output(result_dict)

>>>>>>> feature/optimization-and-refactor
        print("  [ Full JSON output ]")
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))


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

    for resume_file in resume_files:
        result = pipeline.parse(resume_file)
        
        # Basic structural assertions
        assert isinstance(result, dict), "Result must be a dict"
        assert "identity" in result, "Result must have 'identity' context"
        assert "features" in result, "Result must have 'features' for ML"

        name = result["identity"].get("name")
        assert name is not None, "Identity must have 'name'"

        feats = result["features"]
        assert isinstance(feats, dict), "features must be a dict"
        assert len(feats) >= 150, f"Expected 150+ features, got {len(feats)}"

        # Score sanity
        readiness = feats.get("candidate_readiness_score", -1)
        assert 0 <= readiness <= 100, f"Readiness score out of range: {readiness}"

        # ATS penalty sanity
        total_penalty = feats.get("ats_total_penalty_score", -1)
        assert total_penalty >= 0, f"Penalty score must be ≥ 0, got {total_penalty}"

        print(f"  ✅  {resume_file.name}  →  readiness={readiness:.1f}, penalty={total_penalty}")


if __name__ == "__main__":
    main()
