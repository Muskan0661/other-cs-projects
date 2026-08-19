"""
matcher.py
Job matching with heuristics and scoring.
"""

import math
from data import education_levels


class JobMatcher:
    def __init__(self, resume_features):
        """
        Initialize matcher with extracted resume features.

        Parameters
        ----------
        resume_features : dict
            skills              : list[str]
            experience_years    : int
            education_level     : str
            projects_count      : int
            certifications_count: int
        """
        self.resume = resume_features
        self.match_weights = {
            "skills":        0.50,
            "experience":    0.25,
            "education":     0.15,
            "certifications": 0.05,
            "projects":      0.05,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Individual component scores  (all return 0.0 – 1.0)
    # ─────────────────────────────────────────────────────────────────────

    def calculate_skill_match(self, job_skills):
        """Fraction of required job skills present in resume."""
        if not job_skills:
            return 1.0
        candidate_skills_lower = {s.lower() for s in self.resume.get("skills", [])}
        matched = sum(
            1 for s in job_skills
            if s.lower() in candidate_skills_lower
        )
        return matched / len(job_skills)

    def calculate_experience_match(self, required_years):
        """
        Experience score:
        - meets/exceeds requirement → 1.0
        - below requirement         → linear decay (min 0.0)
        """
        candidate_years = self.resume.get("experience_years", 0)
        if required_years == 0:
            return 1.0
        if candidate_years >= required_years:
            return 1.0
        return max(0.0, candidate_years / required_years)

    def calculate_education_match(self, required_education):
        """Education score: 1.0 if meets or exceeds, partial otherwise."""
        candidate_level  = education_levels.get(self.resume.get("education_level", "High School"), 1)
        required_level   = education_levels.get(required_education, 1)
        if candidate_level >= required_level:
            return 1.0
        return max(0.0, candidate_level / required_level)

    def calculate_certifications_score(self):
        """Bonus score for certifications (caps at 1.0 for 3+)."""
        count = self.resume.get("certifications_count", 0)
        return min(1.0, count / 3.0)

    def calculate_projects_score(self):
        """Bonus score for projects (caps at 1.0 for 5+)."""
        count = self.resume.get("projects_count", 0)
        return min(1.0, count / 5.0)

    # ─────────────────────────────────────────────────────────────────────
    # Composite match
    # ─────────────────────────────────────────────────────────────────────

    def calculate_match_score(self, job):
        """
        Compute a weighted composite match score for a single job.

        Returns a dict with all sub-scores and a final match_score (0–100).
        """
        job_skills  = job.get("required_skills", [])
        req_exp     = job.get("required_experience", 0)
        req_edu     = job.get("required_education", "High School")

        skill_score  = self.calculate_skill_match(job_skills)
        exp_score    = self.calculate_experience_match(req_exp)
        edu_score    = self.calculate_education_match(req_edu)
        cert_score   = self.calculate_certifications_score()
        proj_score   = self.calculate_projects_score()

        composite = (
            skill_score  * self.match_weights["skills"] +
            exp_score    * self.match_weights["experience"] +
            edu_score    * self.match_weights["education"] +
            cert_score   * self.match_weights["certifications"] +
            proj_score   * self.match_weights["projects"]
        )

        candidate_skills_lower = {s.lower() for s in self.resume.get("skills", [])}
        matched_skills  = [s for s in job_skills if s.lower() in candidate_skills_lower]
        missing_skills  = [s for s in job_skills if s.lower() not in candidate_skills_lower]

        return {
            "match_score":             round(composite * 100, 2),
            "skill_match_percentage":  round(skill_score * 100, 2),
            "experience_match":        round(exp_score * 100, 2),
            "education_match":         round(edu_score * 100, 2),
            "certifications_score":    round(cert_score * 100, 2),
            "projects_score":          round(proj_score * 100, 2),
            "matched_skills":          matched_skills,
            "missing_skills":          missing_skills,
            "skills_matched_count":    len(matched_skills),
            "skills_required_count":   len(job_skills),
        }

    def rank_jobs(self, jobs):
        """
        Score and rank all jobs.

        Returns a list of dicts sorted by match_score descending:
            [{"job": {...}, "match_info": {...}}, ...]
        """
        results = []
        for job in jobs:
            match_info = self.calculate_match_score(job)
            results.append({"job": job, "match_info": match_info})
        results.sort(key=lambda x: x["match_info"]["match_score"], reverse=True)
        return results
