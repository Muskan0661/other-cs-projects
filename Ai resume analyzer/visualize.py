"""
visualize.py
Visualization and reporting for resume analysis and job recommendations.
"""

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from tabulate import tabulate
    _TABULATE = True
except ImportError:
    _TABULATE = False


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builder
# ─────────────────────────────────────────────────────────────────────────────

def build_recommendation_dataframe(recommendations):
    """Build a pandas DataFrame from recommendations list."""
    if not _PANDAS:
        return None

    data = []
    for rec in recommendations:
        job        = rec["job"]
        match_info = rec["match_info"]
        data.append({
            "Job ID":           job["id"],
            "Title":            job["title"],
            "Company":          job["company"],
            "Location":         job["location"],
            "Match Score (%)":  match_info["match_score"],
            "Skill Match (%)":  match_info["skill_match_percentage"],
            "Exp Match (%)":    match_info["experience_match"],
            "Edu Match (%)":    match_info["education_match"],
            "Matched Skills":   ", ".join(match_info["matched_skills"]),
            "Missing Skills":   ", ".join(match_info["missing_skills"][:5]),
            "Salary Range":     job.get("salary_range", "N/A"),
        })
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Console text report
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendations(recommendations, top_n=5):
    """Print a formatted console report of top recommendations."""
    if not recommendations:
        print("No matching jobs found.")
        return

    print("\n" + "=" * 70)
    print(f"  TOP {min(top_n, len(recommendations))} JOB RECOMMENDATIONS")
    print("=" * 70)

    for rank, rec in enumerate(recommendations[:top_n], start=1):
        job        = rec["job"]
        match_info = rec["match_info"]

        print(f"\n#{rank}  {job['title']}  |  {job['company']}")
        print(f"     Location : {job['location']}")
        print(f"     Salary   : {job.get('salary_range', 'N/A')}")
        print(f"     Match    : {match_info['match_score']:.1f}%  "
              f"(Skills {match_info['skill_match_percentage']:.0f}%  "
              f"Exp {match_info['experience_match']:.0f}%  "
              f"Edu {match_info['education_match']:.0f}%)")

        if match_info["matched_skills"]:
            print(f"     ✓ Skills : {', '.join(match_info['matched_skills'])}")
        if match_info["missing_skills"]:
            print(f"     ✗ Missing: {', '.join(match_info['missing_skills'][:5])}")

    print("\n" + "=" * 70)


def print_resume_summary(resume_meta):
    """Print a summary of extracted resume information."""
    print("\n" + "─" * 50)
    print("  RESUME ANALYSIS SUMMARY")
    print("─" * 50)
    print(f"  Skills Found    : {len(resume_meta.get('raw_skills', []))}")
    print(f"  Experience      : {resume_meta.get('experience_years', {}).get('value', 0)} years")
    print(f"  Education       : {resume_meta.get('education_level', {}).get('value', 'N/A')}")
    print(f"  Projects        : {resume_meta.get('projects_count', {}).get('value', 0)}")
    print(f"  Certifications  : {resume_meta.get('certifications_count', {}).get('value', 0)}")

    skills = resume_meta.get("raw_skills", [])
    if skills:
        print(f"  Detected Skills : {', '.join(skills)}")
    print("─" * 50)


def format_table(recommendations):
    """Return a plain-text table string of recommendations."""
    rows = []
    for rank, rec in enumerate(recommendations, start=1):
        job = rec["job"]
        mi  = rec["match_info"]
        rows.append([
            rank,
            job["title"],
            job["company"],
            f"{mi['match_score']:.1f}%",
            f"{mi['skill_match_percentage']:.0f}%",
            ", ".join(mi["missing_skills"][:3]) or "—",
        ])

    headers = ["#", "Title", "Company", "Match", "Skill Match", "Missing Skills"]

    if _TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline")

    # Fallback plain formatter
    col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    fmt_row = lambda row: "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
    lines = [fmt_row(headers), "-" * sum(col_widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)
