"""
resume_analyzer.py
AI Resume Analyzer - CSP model builder, constraint checker, and NLP extraction.
"""

import re
from data import skill_categories, education_levels


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_resume_model(resume_text):
    """
    Build a structured CSP model from raw resume text.

    Returns
    -------
    variables   : list[str]   – variable names
    domains     : dict        – var -> list of possible values
    resume_meta : dict        – extracted metadata
    """
    variables = []
    domains = {}
    resume_meta = {}

    skills        = extract_skills(resume_text)
    experience    = extract_experience(resume_text)
    education     = extract_education(resume_text)
    projects      = extract_projects(resume_text)
    certifications = extract_certifications(resume_text)

    # Skill variables per category
    for category, cat_skills in skill_categories.items():
        var = f"skill_{category.replace(' ', '_').lower()}"
        variables.append(var)
        found = [s for s in cat_skills if s in skills]
        domains[var] = found if found else ["None"]
        resume_meta[var] = {"category": category, "skills": cat_skills}

    # Experience
    variables.append("experience_years")
    domains["experience_years"] = list(range(0, 21))
    resume_meta["experience_years"] = {"type": "numeric", "value": experience}

    # Education
    variables.append("education_level")
    domains["education_level"] = list(education_levels.keys())
    resume_meta["education_level"] = {"type": "categorical", "value": education}

    # Projects
    variables.append("projects_count")
    domains["projects_count"] = list(range(0, 11))
    resume_meta["projects_count"] = {"type": "numeric", "value": min(len(projects), 10)}

    # Certifications
    variables.append("certifications_count")
    domains["certifications_count"] = list(range(0, 11))
    resume_meta["certifications_count"] = {"type": "numeric", "value": min(len(certifications), 10)}

    # Raw data
    resume_meta["raw_skills"]          = skills
    resume_meta["raw_projects"]        = projects
    resume_meta["raw_certifications"]  = certifications
    resume_meta["raw_text"]            = resume_text

    return variables, domains, resume_meta


def is_consistent(variable, value, assignment, job):
    """
    CSP consistency check – verify that a (variable, value) assignment is
    consistent with job requirements.

    Returns True if the assignment satisfies all relevant constraints.
    """
    if variable == "experience_years":
        return value >= job.get("required_experience", 0)

    if variable == "education_level":
        candidate_rank = education_levels.get(value, 0)
        required_rank  = education_levels.get(job.get("required_education", "High School"), 0)
        return candidate_rank >= required_rank

    if variable.startswith("skill_"):
        # Skills are evaluated together; individual variable always consistent
        return True

    return True


def extract_features_from_meta(resume_meta):
    """
    Flatten resume_meta into a flat feature dict for the matcher.
    """
    features = {
        "skills":               resume_meta.get("raw_skills", []),
        "experience_years":     resume_meta.get("experience_years", {}).get("value", 0),
        "education_level":      resume_meta.get("education_level", {}).get("value", "High School"),
        "projects_count":       resume_meta.get("projects_count", {}).get("value", 0),
        "certifications_count": resume_meta.get("certifications_count", {}).get("value", 0),
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(text):
    """Extract technical skills from resume text."""
    text_lower = text.lower()
    found = set()
    all_skills = [s for cats in skill_categories.values() for s in cats]
    for skill in all_skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return list(found)


def extract_experience(text):
    """Extract maximum years of experience mentioned in text."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'(\d+)\+?\s*years?',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years = [int(m) for m in matches if int(m) <= 40]
            if years:
                return max(years)
    return 0


def extract_education(text):
    """Extract highest education level from resume text."""
    text_lower = text.lower()
    # Map keywords to normalised level names
    checks = [
        (["phd", "ph.d", "doctorate", "doctoral"], "PhD"),
        (["master", "m.s.", "m.sc", "msc", "mba", "m.eng"], "Master"),
        (["bachelor", "b.s.", "b.sc", "bsc", "b.e.", "b.tech", "undergraduate"], "Bachelor"),
        (["associate"], "Associate"),
        (["high school", "secondary school", "hs diploma"], "High School"),
    ]
    for keywords, level in checks:
        if any(kw in text_lower for kw in keywords):
            return level
    return "High School"


def extract_projects(text):
    """Extract project names/descriptions from resume text."""
    projects = []
    # Look for common project section headers
    section_pattern = r'(?:projects?|personal\s+projects?|key\s+projects?)[:\s]*\n(.*?)(?=\n[A-Z]|\Z)'
    section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
    if section_match:
        section = section_match.group(1)
        # Each bullet or line item is a project
        items = re.findall(r'[-•*]\s*(.+)', section)
        projects.extend([i.strip() for i in items if i.strip()])

    # Also look for lines that mention "project" keyword
    for line in text.split('\n'):
        if re.search(r'\bproject\b', line, re.IGNORECASE) and line.strip() not in projects:
            projects.append(line.strip())

    return list(dict.fromkeys(projects))[:10]  # deduplicate, cap at 10


def extract_certifications(text):
    """Extract certifications from resume text."""
    certifications = []
    cert_keywords = [
        "certified", "certification", "certificate", "aws certified",
        "google certified", "azure certified", "pmp", "cissp", "cpa",
        "comptia", "oracle certified", "cisco certified", "ccna", "ccnp"
    ]
    text_lower = text.lower()
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(kw in line_lower for kw in cert_keywords):
            cert = line.strip()
            if cert and cert not in certifications:
                certifications.append(cert)
    return certifications[:10]
