# pyre-ignore-all-errors
"""
synthetic_dataset.py
====================
Generates realistic synthetic resumes with 165 features and proper labels.

Label encoding
--------------
  0 = Weak      (candidate_readiness_score < 45)
  1 = Average   (readiness 45-70)
  2 = Strong    (readiness > 70)

The labels are derived from the same composite score formula used
by feature_extractor.py (G11), ensuring training data is consistent
with inference logic.
"""

from __future__ import annotations

import random
from typing import Dict, List, Any

from .dataset_builder import DATASET_BUILDER
from .dataset_writer import DATASET_WRITER


class SyntheticDatasetGenerator:
    """
    Generates statistically realistic resume feature vectors.

    Each synthetic resume reflects three candidate archetypes:
      - Strong  (30%): high skills, projects, github, good CGPA, experience
      - Average (45%): moderate skills, some projects, basic online presence
      - Weak    (25%): few skills, no projects, no online presence, penalties
    """

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _rand(lo: float, hi: float) -> float:
        # Avoid Pyre round() warnings by using float format
        val = random.uniform(lo, hi)
        return float(f"{val:.4f}")

    @staticmethod
    def _choice(*vals: Any) -> Any:
        return random.choice(vals)

    @staticmethod
    def _bernoulli(p: float) -> int:
        return 1 if random.random() < p else 0

    # ──────────────────────────────────────────────────────
    # Core generators per archetype
    # ──────────────────────────────────────────────────────

    def _strong_resume(self) -> Dict[str, Any]:
        skills = random.randint(12, 22)
        projects = random.randint(3, 6)
        exp_years = self._rand(0.5, 3.0)
        cgpa = self._rand(8.0, 10.0)
        skill_cats = random.randint(4, 8)

        r: Dict[str, Any] = {}
        # G01
        r["has_name"] = 1; r["has_email"] = 1; r["has_phone"] = 1
        r["has_location"] = 1; r["contact_completeness_score"] = 1.0
        # G02
        r["section_count"] = random.randint(6, 9)
        r["has_skills_section"] = 1; r["has_education_section"] = 1
        r["has_experience_section"] = 1; r["has_projects_section"] = 1
        r["has_achievements_section"] = 1; r["has_languages_section"] = 1
        r["has_objective_section"] = self._bernoulli(0.7)
        r["has_interests_section"] = self._bernoulli(0.3)
        r["has_declaration_section"] = 0
        r["section_completeness_score"] = 1.0
        # G03
        r["skills_count"] = skills; r["programming_languages_count"] = random.randint(4, 8)
        r["framework_count"] = random.randint(2, 6); r["database_count"] = random.randint(1, 3)
        r["tool_count"] = random.randint(3, 8)
        r["has_cloud_skills"] = self._bernoulli(0.7); r["has_ai_ml_skills"] = self._bernoulli(0.6)
        r["has_web_dev_skills"] = self._bernoulli(0.8); r["has_mobile_skills"] = self._bernoulli(0.5)
        r["has_devops_skills"] = self._bernoulli(0.6); r["has_security_skills"] = self._bernoulli(0.4)
        r["has_data_skills"] = self._bernoulli(0.5); r["has_testing_skills"] = self._bernoulli(0.4)
        r["skill_category_count"] = skill_cats
        r["skill_weight_score"] = self._rand(18.0, 30.0)
        r["has_high_value_skills"] = 1; r["skill_versatility"] = 1
        r["skill_density"] = self._rand(0.035, 0.065)
        r["project_tech_stack_count"] = random.randint(4, 10)
        r["has_ml_skills"] = r["has_ai_ml_skills"]; r["has_cloud_devops"] = r["has_cloud_skills"]
        r["has_full_stack"] = self._bernoulli(0.6); r["has_ds_skills"] = r["has_data_skills"]
        r["has_mobile_dev"] = r["has_mobile_skills"]
        # G04
        r["education_count"] = random.randint(2, 3)
        r["has_bachelor_degree"] = 1; r["has_master_degree"] = self._bernoulli(0.3)
        r["has_diploma"] = self._bernoulli(0.4); r["has_phd"] = 0
        r["has_btech_be"] = self._bernoulli(0.7); r["has_bca"] = self._bernoulli(0.2)
        r["has_bsc_it"] = self._bernoulli(0.2); r["has_mca"] = self._bernoulli(0.15)
        r["has_mtech_me"] = self._bernoulli(0.15); r["has_msc_it"] = self._bernoulli(0.1)
        r["has_it_major"] = 1; r["is_cs_it_candidate"] = 1
        r["has_top_institution"] = self._bernoulli(0.3)
        r["has_cgpa"] = 1; r["cgpa_value"] = float(f"{cgpa:.2f}")
        r["has_strong_academics"] = int(cgpa >= 8.0)
        r["has_dual_qualification"] = self._bernoulli(0.5)
        r["has_relevant_coursework"] = self._bernoulli(0.4)
        r["graduation_year"] = random.randint(2022, 2025)
        r["is_recent_graduate"] = 1; r["total_skills"] = skills
        # G05
        r["experience_lines"] = random.randint(8, 20)
        r["has_experience"] = 1; r["experience_has_internship"] = 1
        r["experience_years_estimate"] = float(f"{exp_years:.1f}")
        r["internship_role_count"] = random.randint(1, 3)
        r["has_full_time_experience"] = self._bernoulli(0.4)
        r["has_freelance_experience"] = self._bernoulli(0.3)
        r["has_mnc_experience"] = self._bernoulli(0.25)
        r["has_startup_experience"] = self._bernoulli(0.3)
        r["has_remote_experience"] = self._bernoulli(0.4)
        r["has_management_experience"] = 0; r["has_quantified_impact_in_exp"] = self._bernoulli(0.8)
        r["has_missing_dates_in_exp"] = 0; r["experience_company_count"] = random.randint(1, 3)
        r["is_fresher"] = int(exp_years < 0.3)
        r["has_internship_letter"] = 0; r["has_industry_experience"] = self._bernoulli(0.5)
        r["has_research_experience"] = self._bernoulli(0.2)
        # G06
        r["has_projects"] = 1; r["projects_count"] = projects
        r["has_ml_project"] = self._bernoulli(0.5); r["has_web_project"] = self._bernoulli(0.7)
        r["has_mobile_project"] = self._bernoulli(0.4); r["has_database_project"] = self._bernoulli(0.6)
        r["has_api_project"] = self._bernoulli(0.5); r["has_security_project"] = self._bernoulli(0.3)
        r["has_cloud_project"] = self._bernoulli(0.4); r["has_data_project"] = self._bernoulli(0.4)
        r["project_tech_diversity"] = 1; r["has_deployed_project"] = self._bernoulli(0.6)
        r["has_team_project"] = self._bernoulli(0.7); r["project_domain_diversity"] = self._bernoulli(0.7)
        r["avg_project_desc_length"] = self._rand(20.0, 50.0)
        r["has_github_in_projects"] = self._bernoulli(0.7)
        r["has_live_project_url"] = self._bernoulli(0.5); r["has_open_source_contribution"] = self._bernoulli(0.3)
        # G07
        r["achievement_count"] = random.randint(2, 6); r["has_achievements"] = 1
        r["has_hackathon"] = self._bernoulli(0.7); r["hackathon_count"] = random.randint(1, 4)
        r["has_competition"] = self._bernoulli(0.5); r["has_competitive_coding"] = self._bernoulli(0.4)
        r["has_open_source"] = self._bernoulli(0.4); r["has_certifications"] = self._bernoulli(0.8)
        r["certification_platform_count"] = random.randint(1, 4)
        r["has_merit_or_rank"] = self._bernoulli(0.4); r["has_publication"] = self._bernoulli(0.1)
        r["has_volunteer_work"] = self._bernoulli(0.3); r["has_leadership_role"] = self._bernoulli(0.4)
        r["has_workshop_seminar"] = self._bernoulli(0.5)
        # G08
        r["has_linkedin"] = 1; r["has_github"] = 1; r["has_github_profile"] = 1
        r["has_portfolio"] = self._bernoulli(0.4); r["online_presence_count"] = random.randint(2, 3)
        r["online_presence_score"] = self._rand(0.67, 1.0); r["has_multiple_online_links"] = 1
        # G09
        r["resume_word_count"] = random.randint(380, 650)
        r["resume_line_count"] = random.randint(55, 90)
        r["has_bullet_points"] = 1; r["bullet_count"] = random.randint(8, 20)
        r["numbers_count"] = random.randint(10, 30); r["url_count"] = random.randint(1, 4)
        r["is_length_optimal"] = 1; r["is_too_short"] = 0; r["is_too_long"] = 0
        r["action_verb_count"] = random.randint(5, 15); r["has_action_verbs"] = 1
        r["quantified_impact_count"] = random.randint(2, 8); r["has_quantified_impact"] = 1
        r["has_soft_skills"] = 1; r["soft_skill_count"] = random.randint(2, 5)
        r["languages_count"] = random.randint(2, 4); r["extra_language_count"] = random.randint(0, 2)
        r["avg_section_length"] = self._rand(8.0, 18.0); r["generic_buzzword_count"] = random.randint(0, 2)
        r["speaks_multiple_langs"] = 1
        # G10 — mostly clean
        r["ats_penalty_no_skills"] = 0; r["ats_penalty_no_education"] = 0
        r["ats_penalty_no_experience_or_projects"] = 0; r["ats_penalty_no_contact"] = 0
        r["ats_penalty_personal_info"] = 0; r["ats_penalty_unprofessional_email"] = 0
        r["ats_penalty_no_quantified_impact"] = 0; r["ats_penalty_no_action_verbs"] = 0
        r["ats_penalty_missing_dates"] = 0; r["ats_penalty_too_short"] = 0
        r["ats_penalty_too_long"] = 0; r["ats_penalty_buzzword_heavy"] = 0
        r["ats_penalty_spaced_letters"] = 0; r["ats_penalty_keyword_stuffing"] = 0
        r["ats_penalty_no_online_presence"] = 0; r["ats_penalty_has_declaration"] = 0
        r["ats_penalty_skills_has_dates"] = 0; r["ats_penalty_no_proof_of_work"] = 0
        r["ats_total_penalty_score"] = 0; r["penalty_deduction"] = 0.0
        # G11 — compute composite
        pos = (1.0*10 + 1.0*20 + min(r["skill_weight_score"], 30.0) + 8.0 + 8.0 + 8.0 + 5.0 +
               r["has_linkedin"]*4.0 + r["has_github"]*4.0 + 3.0)
        r["raw_positive_score"] = float(f"{min(pos, 100.0):.2f}")
        r["candidate_readiness_score"] = r["raw_positive_score"]
        s_s = min(r["skill_weight_score"]*2 + r["skill_category_count"]*3 + 10, 100.0)
        r["skills_subscore"] = float(f"{s_s:.2f}")
        e_s = min(20 + r["has_strong_academics"]*15 + r["has_top_institution"]*20 + 10, 100.0)
        r["education_subscore"] = float(f"{e_s:.2f}")
        ex_s = min(45 + r["has_quantified_impact_in_exp"]*15 + min(exp_years,3)*10, 100.0)
        r["experience_subscore"] = float(f"{ex_s:.2f}")
        p_s = min(projects*8 + r["project_tech_stack_count"]*3 + r["has_deployed_project"]*15 + r["project_domain_diversity"]*5, 100.0)
        r["projects_subscore"] = float(f"{p_s:.2f}")
        ov = min(r["education_subscore"]*0.20 + r["skills_subscore"]*0.25 + r["experience_subscore"]*0.30 + r["projects_subscore"]*0.15 + r["achievement_count"]*2 + 2 + r["has_github"]*3, 100.0)
        r["overall_profile_strength"] = float(f"{ov:.2f}")
        return r

    def _average_resume(self) -> Dict[str, Any]:
        skills = random.randint(6, 13)
        projects = random.randint(1, 4)
        exp_years = self._rand(0.0, 1.0)
        cgpa = self._rand(6.5, 8.5)
        skill_cats = random.randint(2, 5)

        r: Dict[str, Any] = {}
        r["has_name"] = 1; r["has_email"] = 1; r["has_phone"] = 1
        r["has_location"] = self._bernoulli(0.7); r["contact_completeness_score"] = self._rand(0.75, 1.0)
        r["section_count"] = random.randint(5, 7)
        r["has_skills_section"] = 1; r["has_education_section"] = 1
        r["has_experience_section"] = self._bernoulli(0.7); r["has_projects_section"] = 1
        r["has_achievements_section"] = self._bernoulli(0.7); r["has_languages_section"] = self._bernoulli(0.6)
        r["has_objective_section"] = self._bernoulli(0.5); r["has_interests_section"] = self._bernoulli(0.2)
        r["has_declaration_section"] = self._bernoulli(0.1)
        r["section_completeness_score"] = self._rand(0.5, 1.0)
        r["skills_count"] = skills; r["programming_languages_count"] = random.randint(2, 6)
        r["framework_count"] = random.randint(0, 3); r["database_count"] = random.randint(0, 2)
        r["tool_count"] = random.randint(0, 4)
        r["has_cloud_skills"] = self._bernoulli(0.3); r["has_ai_ml_skills"] = self._bernoulli(0.25)
        r["has_web_dev_skills"] = self._bernoulli(0.6); r["has_mobile_skills"] = self._bernoulli(0.35)
        r["has_devops_skills"] = self._bernoulli(0.2); r["has_security_skills"] = self._bernoulli(0.3)
        r["has_data_skills"] = self._bernoulli(0.2); r["has_testing_skills"] = self._bernoulli(0.2)
        r["skill_category_count"] = skill_cats
        r["skill_weight_score"] = self._rand(7.0, 18.0)
        r["has_high_value_skills"] = self._bernoulli(0.5); r["skill_versatility"] = self._bernoulli(0.5)
        r["skill_density"] = self._rand(0.015, 0.04)
        r["project_tech_stack_count"] = random.randint(1, 5)
        r["has_ml_skills"] = r["has_ai_ml_skills"]; r["has_cloud_devops"] = r["has_cloud_skills"]
        r["has_full_stack"] = self._bernoulli(0.3); r["has_ds_skills"] = r["has_data_skills"]
        r["has_mobile_dev"] = r["has_mobile_skills"]
        r["education_count"] = random.randint(1, 2)
        r["has_bachelor_degree"] = 1; r["has_master_degree"] = 0
        r["has_diploma"] = self._bernoulli(0.3); r["has_phd"] = 0
        r["has_btech_be"] = self._bernoulli(0.6); r["has_bca"] = self._bernoulli(0.2)
        r["has_bsc_it"] = self._bernoulli(0.2); r["has_mca"] = 0; r["has_mtech_me"] = 0; r["has_msc_it"] = 0
        r["has_it_major"] = self._bernoulli(0.8); r["is_cs_it_candidate"] = self._bernoulli(0.8)
        r["has_top_institution"] = 0; r["has_cgpa"] = self._bernoulli(0.8); r["cgpa_value"] = float(f"{cgpa:.2f}")
        r["has_strong_academics"] = int(cgpa >= 8.0); r["has_dual_qualification"] = self._bernoulli(0.3)
        r["has_relevant_coursework"] = self._bernoulli(0.3)
        r["graduation_year"] = random.randint(2022, 2025); r["is_recent_graduate"] = 1; r["total_skills"] = skills
        r["experience_lines"] = random.randint(3, 10); r["has_experience"] = int(exp_years > 0)
        r["experience_has_internship"] = self._bernoulli(0.5)
        r["experience_years_estimate"] = float(f"{exp_years:.1f}")
        r["internship_role_count"] = self._bernoulli(0.5)
        r["has_full_time_experience"] = 0; r["has_freelance_experience"] = self._bernoulli(0.2)
        r["has_mnc_experience"] = 0; r["has_startup_experience"] = self._bernoulli(0.15)
        r["has_remote_experience"] = self._bernoulli(0.2); r["has_management_experience"] = 0
        r["has_quantified_impact_in_exp"] = self._bernoulli(0.3)
        r["has_missing_dates_in_exp"] = self._bernoulli(0.2); r["experience_company_count"] = random.randint(0, 2)
        r["is_fresher"] = int(exp_years < 0.3)
        r["has_internship_letter"] = 0; r["has_industry_experience"] = self._bernoulli(0.3)
        r["has_research_experience"] = 0
        r["has_projects"] = 1; r["projects_count"] = projects
        r["has_ml_project"] = self._bernoulli(0.2); r["has_web_project"] = self._bernoulli(0.5)
        r["has_mobile_project"] = self._bernoulli(0.25); r["has_database_project"] = self._bernoulli(0.5)
        r["has_api_project"] = self._bernoulli(0.3); r["has_security_project"] = self._bernoulli(0.15)
        r["has_cloud_project"] = self._bernoulli(0.15); r["has_data_project"] = self._bernoulli(0.2)
        r["project_tech_diversity"] = self._bernoulli(0.5); r["has_deployed_project"] = self._bernoulli(0.2)
        r["has_team_project"] = self._bernoulli(0.4); r["project_domain_diversity"] = self._bernoulli(0.3)
        r["avg_project_desc_length"] = self._rand(10.0, 25.0); r["has_github_in_projects"] = self._bernoulli(0.3)
        r["has_live_project_url"] = self._bernoulli(0.15); r["has_open_source_contribution"] = self._bernoulli(0.1)
        r["achievement_count"] = random.randint(1, 3); r["has_achievements"] = self._bernoulli(0.7)
        r["has_hackathon"] = self._bernoulli(0.4); r["hackathon_count"] = random.randint(0, 2)
        r["has_competition"] = self._bernoulli(0.2); r["has_competitive_coding"] = self._bernoulli(0.2)
        r["has_open_source"] = self._bernoulli(0.1); r["has_certifications"] = self._bernoulli(0.5)
        r["certification_platform_count"] = random.randint(0, 2)
        r["has_merit_or_rank"] = self._bernoulli(0.2); r["has_publication"] = 0
        r["has_volunteer_work"] = self._bernoulli(0.15); r["has_leadership_role"] = self._bernoulli(0.2)
        r["has_workshop_seminar"] = self._bernoulli(0.3)
        r["has_linkedin"] = self._bernoulli(0.8); r["has_github"] = self._bernoulli(0.5)
        r["has_github_profile"] = r["has_github"]; r["has_portfolio"] = self._bernoulli(0.15)
        r["online_presence_count"] = r["has_linkedin"] + r["has_github"] + r["has_portfolio"]
        r["online_presence_score"] = float(f"{(r['online_presence_count'] / 3.0):.2f}")
        r["has_multiple_online_links"] = int(r["online_presence_count"] >= 2)
        r["resume_word_count"] = random.randint(270, 500); r["resume_line_count"] = random.randint(40, 70)
        r["has_bullet_points"] = self._bernoulli(0.7); r["bullet_count"] = random.randint(2, 10)
        r["numbers_count"] = random.randint(4, 18); r["url_count"] = random.randint(0, 2)
        r["is_length_optimal"] = 1; r["is_too_short"] = 0; r["is_too_long"] = 0
        r["action_verb_count"] = random.randint(0, 5); r["has_action_verbs"] = self._bernoulli(0.5)
        r["quantified_impact_count"] = random.randint(0, 2); r["has_quantified_impact"] = self._bernoulli(0.4)
        r["has_soft_skills"] = self._bernoulli(0.7); r["soft_skill_count"] = random.randint(0, 3)
        r["languages_count"] = random.randint(1, 3); r["extra_language_count"] = random.randint(0, 1)
        r["avg_section_length"] = self._rand(5.0, 12.0); r["generic_buzzword_count"] = random.randint(0, 3)
        r["speaks_multiple_langs"] = int(r["languages_count"] >= 2)
        # ATS penalties — some present
        r["ats_penalty_no_skills"] = 0; r["ats_penalty_no_education"] = 0
        r["ats_penalty_no_experience_or_projects"] = 0; r["ats_penalty_no_contact"] = 0
        r["ats_penalty_personal_info"] = self._bernoulli(0.1)
        r["ats_penalty_unprofessional_email"] = self._bernoulli(0.05)
        r["ats_penalty_no_quantified_impact"] = int(not r["has_quantified_impact"])
        r["ats_penalty_no_action_verbs"] = int(not r["has_action_verbs"])
        r["ats_penalty_missing_dates"] = self._bernoulli(0.2)
        r["ats_penalty_too_short"] = 0; r["ats_penalty_too_long"] = 0
        r["ats_penalty_buzzword_heavy"] = self._bernoulli(0.15)
        r["ats_penalty_spaced_letters"] = self._bernoulli(0.1)
        r["ats_penalty_keyword_stuffing"] = 0
        r["ats_penalty_no_online_presence"] = int(r["online_presence_count"] == 0)
        r["ats_penalty_has_declaration"] = self._bernoulli(0.1)
        r["ats_penalty_skills_has_dates"] = self._bernoulli(0.2)
        r["ats_penalty_no_proof_of_work"] = 0
        penalty_keys = [k for k in r if k.startswith("ats_penalty_") and k != "ats_total_penalty_score" and r[k] == 1]
        r["ats_total_penalty_score"] = len(penalty_keys)
        r["penalty_deduction"] = float(len(penalty_keys)) * 4.0
        # G11
        pos = (r["contact_completeness_score"]*10.0 + r["section_completeness_score"]*20.0 +
               min(r["skill_weight_score"], 30.0) + r["has_high_value_skills"]*8.0 +
               r["has_projects"]*8.0 + r["has_experience"]*8.0 + r["has_achievements"]*5.0 +
               r["has_linkedin"]*4.0 + r["has_github"]*4.0 + r["has_cgpa"]*3.0)
        r["raw_positive_score"] = float(f"{min(pos, 100.0):.2f}")
        r["candidate_readiness_score"] = float(f"{max(0.0, min(r['raw_positive_score'] - r['penalty_deduction'], 100.0)):.2f}")
        s_s = min(r["skill_weight_score"]*2.0 + r["skill_category_count"]*3.0 + r["has_high_value_skills"]*10.0, 100.0)
        r["skills_subscore"] = float(f"{s_s:.2f}")
        e_s = min(r["has_bachelor_degree"]*20.0 + r["has_master_degree"]*30.0 + r["has_strong_academics"]*15.0 + r["has_top_institution"]*20.0 + r["has_cgpa"]*10.0, 100.0)
        r["education_subscore"] = float(f"{e_s:.2f}")
        ex_s = min(r["has_experience"]*25.0 + r["experience_has_internship"]*20.0 + r["has_quantified_impact_in_exp"]*15.0 + min(exp_years, 3)*10.0, 100.0)
        r["experience_subscore"] = float(f"{ex_s:.2f}")
        p_s = min(projects*8.0 + r["project_tech_stack_count"]*3.0 + r["has_deployed_project"]*15.0 + r["project_domain_diversity"]*5.0, 100.0)
        r["projects_subscore"] = float(f"{p_s:.2f}")
        ov = min(r["education_subscore"]*0.20 + r["skills_subscore"]*0.25 + r["experience_subscore"]*0.30 + r["projects_subscore"]*0.15 + r["achievement_count"]*2.0 + r["has_linkedin"]*2.0 + r["has_github"]*3.0, 100.0)
        r["overall_profile_strength"] = float(f"{ov:.2f}")
        return r

    def _weak_resume(self) -> Dict[str, Any]:
        skills = random.randint(1, 7)
        projects = random.randint(0, 2)
        cgpa = self._rand(4.0, 7.0)
        r: Dict[str, Any] = {}
        r["has_name"] = 1; r["has_email"] = 1; r["has_phone"] = self._bernoulli(0.7)
        r["has_location"] = self._bernoulli(0.5); r["contact_completeness_score"] = self._rand(0.5, 0.85)
        r["section_count"] = random.randint(2, 5)
        r["has_skills_section"] = self._bernoulli(0.7); r["has_education_section"] = 1
        r["has_experience_section"] = self._bernoulli(0.3); r["has_projects_section"] = self._bernoulli(0.4)
        r["has_achievements_section"] = self._bernoulli(0.2); r["has_languages_section"] = self._bernoulli(0.3)
        r["has_objective_section"] = self._bernoulli(0.4); r["has_interests_section"] = self._bernoulli(0.1)
        r["has_declaration_section"] = self._bernoulli(0.4)
        r["section_completeness_score"] = self._rand(0.1, 0.5)
        r["skills_count"] = skills; r["programming_languages_count"] = random.randint(0, 3)
        r["framework_count"] = 0; r["database_count"] = random.randint(0, 1); r["tool_count"] = 0
        r["has_cloud_skills"] = 0; r["has_ai_ml_skills"] = 0; r["has_web_dev_skills"] = self._bernoulli(0.3)
        r["has_mobile_skills"] = 0; r["has_devops_skills"] = 0
        r["has_security_skills"] = 0; r["has_data_skills"] = 0; r["has_testing_skills"] = 0
        r["skill_category_count"] = random.randint(0, 2); r["skill_weight_score"] = self._rand(0.0, 7.0)
        r["has_high_value_skills"] = 0; r["skill_versatility"] = 0; r["skill_density"] = self._rand(0.005, 0.02)
        r["project_tech_stack_count"] = random.randint(0, 2)
        r["has_ml_skills"] = 0; r["has_cloud_devops"] = 0; r["has_full_stack"] = 0
        r["has_ds_skills"] = 0; r["has_mobile_dev"] = 0
        r["education_count"] = 1; r["has_bachelor_degree"] = 1; r["has_master_degree"] = 0
        r["has_diploma"] = self._bernoulli(0.2); r["has_phd"] = 0
        r["has_btech_be"] = self._bernoulli(0.4); r["has_bca"] = self._bernoulli(0.3)
        r["has_bsc_it"] = self._bernoulli(0.3); r["has_mca"] = 0; r["has_mtech_me"] = 0; r["has_msc_it"] = 0
        r["has_it_major"] = self._bernoulli(0.5); r["is_cs_it_candidate"] = self._bernoulli(0.5)
        r["has_top_institution"] = 0; r["has_cgpa"] = self._bernoulli(0.5); r["cgpa_value"] = float(f"{cgpa:.2f}")
        r["has_strong_academics"] = 0; r["has_dual_qualification"] = 0; r["has_relevant_coursework"] = 0
        r["graduation_year"] = random.randint(2020, 2025); r["is_recent_graduate"] = self._bernoulli(0.5)
        r["total_skills"] = skills
        r["experience_lines"] = random.randint(0, 5); r["has_experience"] = 0
        r["experience_has_internship"] = 0; r["experience_years_estimate"] = 0.0
        r["internship_role_count"] = 0; r["has_full_time_experience"] = 0; r["has_freelance_experience"] = 0
        r["has_mnc_experience"] = 0; r["has_startup_experience"] = 0; r["has_remote_experience"] = 0
        r["has_management_experience"] = 0; r["has_quantified_impact_in_exp"] = 0
        r["has_missing_dates_in_exp"] = self._bernoulli(0.4); r["experience_company_count"] = 0
        r["is_fresher"] = 1; r["has_internship_letter"] = 0; r["has_industry_experience"] = 0
        r["has_research_experience"] = 0
        r["has_projects"] = int(projects > 0); r["projects_count"] = projects
        r["has_ml_project"] = 0; r["has_web_project"] = self._bernoulli(0.2)
        r["has_mobile_project"] = 0; r["has_database_project"] = self._bernoulli(0.2)
        r["has_api_project"] = 0; r["has_security_project"] = 0; r["has_cloud_project"] = 0; r["has_data_project"] = 0
        r["project_tech_diversity"] = 0; r["has_deployed_project"] = 0; r["has_team_project"] = 0
        r["project_domain_diversity"] = 0; r["avg_project_desc_length"] = self._rand(3.0, 12.0)
        r["has_github_in_projects"] = 0; r["has_live_project_url"] = 0; r["has_open_source_contribution"] = 0
        r["achievement_count"] = random.randint(0, 1); r["has_achievements"] = self._bernoulli(0.3)
        r["has_hackathon"] = 0; r["hackathon_count"] = 0; r["has_competition"] = 0
        r["has_competitive_coding"] = 0; r["has_open_source"] = 0; r["has_certifications"] = self._bernoulli(0.2)
        r["certification_platform_count"] = 0; r["has_merit_or_rank"] = 0; r["has_publication"] = 0
        r["has_volunteer_work"] = 0; r["has_leadership_role"] = 0; r["has_workshop_seminar"] = 0
        r["has_linkedin"] = self._bernoulli(0.4); r["has_github"] = self._bernoulli(0.2)
        r["has_github_profile"] = r["has_github"]; r["has_portfolio"] = 0
        r["online_presence_count"] = r["has_linkedin"] + r["has_github"]
        r["online_presence_score"] = float(f"{(r['online_presence_count'] / 3.0):.2f}")
        r["has_multiple_online_links"] = int(r["online_presence_count"] >= 2)
        r["resume_word_count"] = random.randint(80, 280); r["resume_line_count"] = random.randint(15, 45)
        r["has_bullet_points"] = self._bernoulli(0.3); r["bullet_count"] = random.randint(0, 3)
        r["numbers_count"] = random.randint(0, 6); r["url_count"] = 0
        r["is_length_optimal"] = 0; r["is_too_short"] = 1; r["is_too_long"] = 0
        r["action_verb_count"] = 0; r["has_action_verbs"] = 0; r["quantified_impact_count"] = 0
        r["has_quantified_impact"] = 0; r["has_soft_skills"] = self._bernoulli(0.3); r["soft_skill_count"] = 0
        r["languages_count"] = random.randint(1, 2); r["extra_language_count"] = 0
        r["avg_section_length"] = self._rand(1.0, 6.0); r["generic_buzzword_count"] = random.randint(1, 5)
        r["speaks_multiple_langs"] = int(r["languages_count"] >= 2)
        # ATS penalties — many present
        r["ats_penalty_no_skills"] = int(skills < 3)
        r["ats_penalty_no_education"] = 0
        r["ats_penalty_no_experience_or_projects"] = int(projects == 0)
        r["ats_penalty_no_contact"] = int(not r["has_phone"])
        r["ats_penalty_personal_info"] = self._bernoulli(0.4)
        r["ats_penalty_unprofessional_email"] = self._bernoulli(0.15)
        r["ats_penalty_no_quantified_impact"] = 1; r["ats_penalty_no_action_verbs"] = 1
        r["ats_penalty_missing_dates"] = self._bernoulli(0.5)
        r["ats_penalty_too_short"] = 1; r["ats_penalty_too_long"] = 0
        r["ats_penalty_buzzword_heavy"] = self._bernoulli(0.3)
        r["ats_penalty_spaced_letters"] = self._bernoulli(0.2)
        r["ats_penalty_keyword_stuffing"] = self._bernoulli(0.1)
        r["ats_penalty_no_online_presence"] = int(r["online_presence_count"] == 0)
        r["ats_penalty_has_declaration"] = self._bernoulli(0.4)
        r["ats_penalty_skills_has_dates"] = self._bernoulli(0.4)
        r["ats_penalty_no_proof_of_work"] = int(projects == 0)
        penalty_keys = [k for k in r if k.startswith("ats_penalty_") and k != "ats_total_penalty_score" and r.get(k) == 1]
        r["ats_total_penalty_score"] = len(penalty_keys)
        r["penalty_deduction"] = float(len(penalty_keys)) * 4.0
        # G11
        pos = (r["contact_completeness_score"]*10.0 + r["section_completeness_score"]*20.0 +
               min(r["skill_weight_score"], 30.0) + r["has_high_value_skills"]*8.0 +
               r["has_projects"]*8.0 + r["has_experience"]*8.0 + r["has_achievements"]*5.0 +
               r["has_linkedin"]*4.0 + r["has_github"]*4.0 + r["has_cgpa"]*3.0)
        r["raw_positive_score"] = float(f"{min(pos, 100.0):.2f}")
        r["candidate_readiness_score"] = float(f"{max(0.0, min(r['raw_positive_score'] - r['penalty_deduction'], 100.0)):.2f}")
        s_s = min(r["skill_weight_score"]*2.0 + r["skill_category_count"]*3.0, 100.0)
        r["skills_subscore"] = float(f"{s_s:.2f}")
        e_s = min(r["has_bachelor_degree"]*20.0 + r["has_cgpa"]*10.0, 100.0)
        r["education_subscore"] = float(f"{e_s:.2f}")
        r["experience_subscore"] = 0.0
        p_s = min(projects*8.0 + r["project_tech_stack_count"]*3.0, 100.0)
        r["projects_subscore"] = float(f"{p_s:.2f}")
        ov = min(r["education_subscore"]*0.20 + r["skills_subscore"]*0.25 + r["experience_subscore"]*0.30 + r["projects_subscore"]*0.15 + r["achievement_count"]*2.0 + r["has_linkedin"]*2.0 + r["has_github"]*3.0, 100.0)
        r["overall_profile_strength"] = float(f"{ov:.2f}")
        return r

    # ──────────────────────────────────────────────────────
    # Label assignment
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _assign_label(row: Dict[str, Any]) -> int:
        """
        Label from candidate_readiness_score (the engine's own composite metric):
          0 = Weak    (<45)
          1 = Average (45–70)
          2 = Strong  (>70)
        """
        score = float(row.get("candidate_readiness_score", 0))
        if score > 70:
            return 2
        if score >= 45:
            return 1
        return 0

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def generate(self, n_samples: int = 2000) -> None:
        """
        Generate *n_samples* synthetic resumes and write to CSV.

        Distribution: 30% strong, 45% average, 25% weak (realistic CS/IT pool).
        """
        n_strong  = int(n_samples * 0.30)
        n_average = int(n_samples * 0.45)
        n_weak    = n_samples - n_strong - n_average

        features: List[Dict[str, Any]] = []
        labels: List[int] = []

        for _ in range(n_strong):
            row = self._strong_resume()
            features.append(row)
            labels.append(self._assign_label(row))

        for _ in range(n_average):
            row = self._average_resume()
            features.append(row)
            labels.append(self._assign_label(row))

        for _ in range(n_weak):
            row = self._weak_resume()
            features.append(row)
            labels.append(self._assign_label(row))

        # Shuffle
        combined = list(zip(features, labels))
        random.shuffle(combined)
        shuffled_features = [x[0] for x in combined]
        shuffled_labels = [x[1] for x in combined]

        rows = DATASET_BUILDER.build_rows(shuffled_features, shuffled_labels)
        from pathlib import Path
        output_dir = Path(__file__).resolve().parent.parent / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "resume_dataset.csv"

        DATASET_WRITER.write_csv(rows, str(output_path))

        label_dist = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
        print(f"✅  Dataset generated: {n_samples} samples")
        print(f"    Weak(0)={label_dist[0]}  Average(1)={label_dist[1]}  Strong(2)={label_dist[2]}")
        print(f"    Saved → {output_path}")