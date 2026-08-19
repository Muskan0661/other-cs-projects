"""
main.py
Command-line entry point for the AI Resume Analyzer.
"""

import sys
from data import load_jobs
from resume_analyzer import build_resume_model, extract_features_from_meta
from matcher import JobMatcher
from solver import solve
from visualize import print_resume_summary, print_recommendations, format_table


SAMPLE_RESUME = """
John Doe
Software Engineer | john.doe@email.com

SUMMARY
Passionate software engineer with 4 years of experience building scalable
web applications and data pipelines.

SKILLS
Python, Django, Flask, JavaScript, React, Node.js, SQL, PostgreSQL,
Docker, Git, AWS, Pandas, NumPy, Scikit-learn, Linux

EXPERIENCE
Software Engineer – TechStartup Inc  (2020 – 2024)
• 4 years of experience developing REST APIs with Python and Django
• Deployed microservices on AWS using Docker and Kubernetes

Junior Developer – WebAgency Co  (2019 – 2020)

EDUCATION
Bachelor of Science in Computer Science
State University, 2019

PROJECTS
• ResumeBot – NLP-powered resume screening tool (Python, Scikit-learn)
• DashKit – real-time analytics dashboard (React, Node.js)
• PipeFlow – ETL pipeline (Python, Pandas, PostgreSQL)

CERTIFICATIONS
AWS Certified Developer – Associate
"""


def run_analysis(resume_text: str, top_n: int = 5):
    """Core analysis pipeline."""
    print("\n🔍 Analyzing resume …")
    variables, domains, resume_meta = build_resume_model(resume_text)
    print_resume_summary(resume_meta)

    resume_features = extract_features_from_meta(resume_meta)
    jobs = load_jobs()
    print(f"\n📋 Loaded {len(jobs)} job postings.")

    matcher = JobMatcher(resume_features)
    recommendations, nodes = solve(jobs, matcher, resume_features, threshold=0, top_n=top_n)
    print(f"⚙️  CSP solver explored {nodes} nodes.")

    print_recommendations(recommendations, top_n=top_n)
    print("\n" + format_table(recommendations))
    return recommendations


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
        try:
            with open(filepath, encoding="utf-8") as f:
                resume_text = f.read()
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            sys.exit(1)
    else:
        print("No resume file supplied – using built-in sample resume.")
        resume_text = SAMPLE_RESUME

    run_analysis(resume_text)
