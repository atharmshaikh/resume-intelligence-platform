"""
feature_schema.py
=================
Canonical ML feature schema — single source of truth for training & inference.

Must stay in sync with ml_engine.features.feature_extractor.extract_features()
output (165 features, groups G01–G11).

Rules
-----
- Feature names are snake_case
- Order here == column order in CSV dataset == model input order
- Never remove features (breaks trained model). Add new ones at the end.
"""

from __future__ import annotations

from typing import List


class FeatureSchemaError(Exception):
    """Raised on schema validation failure."""
    pass


class FeatureSchema:
    """
    Ordered feature registry.

    Guarantees consistent feature order between:
      - dataset generation  (synthetic_dataset.py)
      - model training      (trainer.py)
      - inference           (predictor.py)
    """

    def __init__(self, features: List[str]) -> None:
        if not features:
            raise FeatureSchemaError("Feature list cannot be empty")
        if len(set(features)) != len(features):
            dups = [f for f in features if features.count(f) > 1]
            raise FeatureSchemaError(f"Duplicate features: {set(dups)}")
        self._features: List[str] = list(features)

    def get_features(self) -> List[str]:
        """Return ordered feature list (copy)."""
        return list(self._features)

    def size(self) -> int:
        """Number of features."""
        return len(self._features)

    def validate(self, feature_dict: dict) -> None:
        """Soft validate — dict must be non-empty."""
        if not isinstance(feature_dict, dict):
            raise FeatureSchemaError("Feature input must be a dictionary")

    def default_row(self) -> dict:
        """Return a zero-filled feature dict (safe fallback)."""
        return {f: 0 for f in self._features}

    def align(self, raw: dict) -> dict:
        """
        Align a raw feature dict to schema order.
        Missing keys → 0, extra keys → ignored.
        """
        return {f: (raw.get(f) or 0) for f in self._features}


# =============================================================================
# FEATURE LIST  (165 features — matches feature_extractor.py G01–G11)
# =============================================================================

FEATURE_LIST: List[str] = [

    # ── G01 · Contact completeness (5) ───────────────────────────────────────
    "has_name",
    "has_email",
    "has_phone",
    "has_location",
    "contact_completeness_score",

    # ── G02 · Section structure (11) ─────────────────────────────────────────
    "section_count",
    "has_skills_section",
    "has_education_section",
    "has_experience_section",
    "has_projects_section",
    "has_achievements_section",
    "has_languages_section",
    "has_objective_section",
    "has_interests_section",
    "has_declaration_section",
    "section_completeness_score",

    # ── G03 · Skill taxonomy (24) ────────────────────────────────────────────
    "skills_count",
    "programming_languages_count",
    "framework_count",
    "database_count",
    "tool_count",
    "has_cloud_skills",
    "has_ai_ml_skills",
    "has_web_dev_skills",
    "has_mobile_skills",
    "has_devops_skills",
    "has_security_skills",
    "has_data_skills",
    "has_testing_skills",
    "skill_category_count",
    "skill_weight_score",
    "has_high_value_skills",
    "skill_versatility",
    "skill_density",
    "project_tech_stack_count",
    # sub-domain flags used by feature extractor
    "has_ml_skills",
    "has_cloud_devops",
    "has_full_stack",
    "has_ds_skills",
    "has_mobile_dev",

    # ── G04 · Education depth (22) ───────────────────────────────────────────
    "education_count",
    "has_bachelor_degree",
    "has_master_degree",
    "has_diploma",
    "has_phd",
    "has_btech_be",
    "has_bca",
    "has_bsc_it",
    "has_mca",
    "has_mtech_me",
    "has_msc_it",
    "has_it_major",
    "is_cs_it_candidate",
    "has_top_institution",
    "has_cgpa",
    "cgpa_value",
    "has_strong_academics",
    "has_dual_qualification",
    "has_relevant_coursework",
    "graduation_year",
    "is_recent_graduate",
    "total_skills",

    # ── G05 · Experience analysis (18) ───────────────────────────────────────
    "experience_lines",
    "has_experience",
    "experience_has_internship",
    "experience_years_estimate",
    "internship_role_count",
    "has_full_time_experience",
    "has_freelance_experience",
    "has_mnc_experience",
    "has_startup_experience",
    "has_remote_experience",
    "has_management_experience",
    "has_quantified_impact_in_exp",
    "has_missing_dates_in_exp",
    "experience_company_count",
    "is_fresher",
    # padding to 18
    "has_internship_letter",
    "has_industry_experience",
    "has_research_experience",

    # ── G06 · Projects (18) ──────────────────────────────────────────────────
    "has_projects",
    "projects_count",
    "has_ml_project",
    "has_web_project",
    "has_mobile_project",
    "has_database_project",
    "has_api_project",
    "has_security_project",
    "has_cloud_project",
    "has_data_project",
    "project_tech_diversity",
    "has_deployed_project",
    "has_team_project",
    "project_domain_diversity",
    "avg_project_desc_length",
    "has_github_in_projects",
    # padding to 18
    "has_live_project_url",
    "has_open_source_contribution",

    # ── G07 · Achievements, certifications, extras (14) ──────────────────────
    "achievement_count",
    "has_achievements",
    "has_hackathon",
    "hackathon_count",
    "has_competition",
    "has_competitive_coding",
    "has_open_source",
    "has_certifications",
    "certification_platform_count",
    "has_merit_or_rank",
    "has_publication",
    "has_volunteer_work",
    "has_leadership_role",
    "has_workshop_seminar",

    # ── G08 · Online presence (7) ────────────────────────────────────────────
    "has_linkedin",
    "has_github",
    "has_github_profile",
    "has_portfolio",
    "online_presence_count",
    "online_presence_score",
    "has_multiple_online_links",

    # ── G09 · Resume format quality (16) ─────────────────────────────────────
    "resume_word_count",
    "resume_line_count",
    "has_bullet_points",
    "bullet_count",
    "numbers_count",
    "url_count",
    "is_length_optimal",
    "is_too_short",
    "is_too_long",
    "action_verb_count",
    "has_action_verbs",
    "quantified_impact_count",
    "has_quantified_impact",
    "has_soft_skills",
    "soft_skill_count",
    "languages_count",
    # bonus (part of G09)
    "extra_language_count",
    "avg_section_length",
    "generic_buzzword_count",
    "speaks_multiple_langs",

    # ── G10 · ATS penalty flags (20) ─────────────────────────────────────────
    "ats_penalty_no_skills",
    "ats_penalty_no_education",
    "ats_penalty_no_experience_or_projects",
    "ats_penalty_no_contact",
    "ats_penalty_personal_info",
    "ats_penalty_unprofessional_email",
    "ats_penalty_no_quantified_impact",
    "ats_penalty_no_action_verbs",
    "ats_penalty_missing_dates",
    "ats_penalty_too_short",
    "ats_penalty_too_long",
    "ats_penalty_buzzword_heavy",
    "ats_penalty_spaced_letters",
    "ats_penalty_keyword_stuffing",
    "ats_penalty_no_online_presence",
    "ats_penalty_has_declaration",
    "ats_penalty_skills_has_dates",
    "ats_penalty_no_proof_of_work",
    "ats_total_penalty_score",
    "penalty_deduction",

    # ── G11 · Composite / derived scores (10) ────────────────────────────────
    "raw_positive_score",
    "candidate_readiness_score",
    "skills_subscore",
    "education_subscore",
    "experience_subscore",
    "projects_subscore",
    "overall_profile_strength",
    "contact_completeness_score",  # intentional alias (also in G11 composite use)
    "section_completeness_score",  # intentional alias
    "skill_weight_score",          # intentional alias

]

# De-duplicate while preserving order (aliases listed twice above)
_seen: set = set()
_deduped: List[str] = []
for _f in FEATURE_LIST:
    if _f not in _seen:
        _seen.add(_f)
        _deduped.append(_f)
FEATURE_LIST = _deduped

FEATURE_SCHEMA = FeatureSchema(FEATURE_LIST)