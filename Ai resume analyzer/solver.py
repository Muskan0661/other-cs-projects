"""
solver.py
Backtracking search with heuristics for job recommendation.
"""

import copy
from resume_analyzer import is_consistent

# Global counter for nodes explored
nodes_explored = 0


def select_best_job(jobs, assignment, matcher):
    """
    Heuristic for selecting the next job to evaluate.
    Prioritise jobs with higher potential match score (MRV-style).
    """
    unassigned = [job for job in jobs if job["id"] not in assignment]
    if not unassigned:
        return None

    # Score each unassigned job and pick the highest
    scored = []
    for job in unassigned:
        match_info = matcher.calculate_match_score(job)
        scored.append((job, match_info["match_score"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def order_values(job, matcher):
    """
    Return a list of threshold scores to try for this job.
    We use a simple descending range so the solver tries high thresholds first.
    """
    return [70, 50, 30, 0]   # minimum acceptable match score thresholds


def backtrack(jobs, assignment, matcher, resume_features, threshold=0, limit=5):
    """
    Backtracking search that fills `assignment` with job_id -> match_info pairs
    for jobs whose match score meets `threshold`.

    Parameters
    ----------
    jobs            : list of job dicts
    assignment      : dict  {job_id: match_info}  (modified in place)
    matcher         : JobMatcher instance
    resume_features : dict
    threshold       : minimum match score to accept a job
    limit           : maximum number of recommendations to return

    Returns a list of accepted recommendation dicts.
    """
    global nodes_explored

    if len(assignment) >= limit:
        return _build_result(assignment, jobs)

    job = select_best_job(jobs, assignment, matcher)
    if job is None:
        return _build_result(assignment, jobs)

    nodes_explored += 1

    match_info = matcher.calculate_match_score(job)

    # Constraint check via is_consistent
    exp_ok = is_consistent(
        "experience_years",
        resume_features.get("experience_years", 0),
        assignment, job
    )
    edu_ok = is_consistent(
        "education_level",
        resume_features.get("education_level", "High School"),
        assignment, job
    )

    score = match_info["match_score"]
    if score >= threshold and exp_ok and edu_ok:
        assignment[job["id"]] = match_info
    else:
        # Mark as visited but rejected (use None)
        assignment[job["id"]] = None

    return backtrack(jobs, assignment, matcher, resume_features, threshold, limit)


def _build_result(assignment, jobs):
    """Convert assignment dict back to recommendation list."""
    job_map = {j["id"]: j for j in jobs}
    results = []
    for job_id, match_info in assignment.items():
        if match_info is not None:
            results.append({
                "job": job_map[job_id],
                "match_info": match_info
            })
    results.sort(key=lambda x: x["match_info"]["match_score"], reverse=True)
    return results


def solve(jobs, matcher, resume_features, threshold=30, top_n=5):
    """
    Main entry point for the CSP solver.

    Returns top-N job recommendations sorted by match score.
    """
    global nodes_explored
    nodes_explored = 0
    assignment = {}
    recommendations = backtrack(
        jobs, assignment, matcher, resume_features,
        threshold=threshold, limit=len(jobs)
    )
    return recommendations[:top_n], nodes_explored
