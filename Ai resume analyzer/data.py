"""
data.py
Default fallback data + CSV loader for resumes and jobs.
If jobs.csv exists in the same folder, it will be loaded automatically.
Otherwise, the hardcoded defaults below are used.
"""

import os
import csv

# ── Default skill categories for extraction ──────────────────────────────
skill_categories = {
    "Programming Languages": ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift"],
    "Web Development": ["React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring", "ASP.NET", "HTML5", "CSS3", "TypeScript"],
    "Data Science": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "SQL", "Tableau", "Statistics", "Data Visualization"],
    "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "CI/CD", "Terraform", "Linux"],
    "Soft Skills": ["Leadership", "Communication", "Teamwork", "Problem Solving", "Project Management", "Agile", "Scrum"]
}

# ── Education level hierarchy ─────────────────────────────────────────────
education_levels = {
    "PhD": 5,
    "Master": 4,
    "Bachelor": 3,
    "Associate": 2,
    "High School": 1
}

# ── Hardcoded fallback jobs ───────────────────────────────────────────────
_default_jobs = [
    {
        "id": 1,
        "title": "Senior Python Developer",
        "company": "TechCorp Solutions",
        "location": "New York, NY",
        "required_skills": ["Python", "Django", "SQL", "Git", "REST APIs"],
        "required_experience": 3,
        "required_education": "Bachelor",
        "salary_range": "$90,000 - $120,000",
        "description": "Looking for an experienced Python developer to build scalable web applications."
    },
    {
        "id": 2,
        "title": "Machine Learning Engineer",
        "company": "AI Innovations Inc",
        "location": "San Francisco, CA",
        "required_skills": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "SQL"],
        "required_experience": 2,
        "required_education": "Master",
        "salary_range": "$120,000 - $160,000",
        "description": "Develop and deploy ML models for production systems."
    },
    {
        "id": 3,
        "title": "Full Stack Developer",
        "company": "WebWorks Agency",
        "location": "Austin, TX",
        "required_skills": ["JavaScript", "React", "Node.js", "SQL", "Git"],
        "required_experience": 2,
        "required_education": "Bachelor",
        "salary_range": "$80,000 - $110,000",
        "description": "Build modern web applications using MERN stack."
    },
    {
        "id": 4,
        "title": "Data Scientist",
        "company": "DataMind Analytics",
        "location": "Seattle, WA",
        "required_skills": ["Python", "Pandas", "NumPy", "Scikit-learn", "SQL", "Tableau"],
        "required_experience": 3,
        "required_education": "Master",
        "salary_range": "$100,000 - $140,000",
        "description": "Analyze complex datasets and build predictive models."
    },
    {
        "id": 5,
        "title": "DevOps Engineer",
        "company": "CloudScale Systems",
        "location": "Remote",
        "required_skills": ["Docker", "Kubernetes", "AWS", "Jenkins", "Linux", "Terraform", "Git"],
        "required_experience": 3,
        "required_education": "Bachelor",
        "salary_range": "$110,000 - $150,000",
        "description": "Manage cloud infrastructure and CI/CD pipelines."
    },
    {
        "id": 6,
        "title": "Frontend Developer",
        "company": "Creative Digital",
        "location": "Los Angeles, CA",
        "required_skills": ["JavaScript", "React", "HTML5", "CSS3", "TypeScript"],
        "required_experience": 1,
        "required_education": "Bachelor",
        "salary_range": "$70,000 - $95,000",
        "description": "Create responsive and interactive user interfaces."
    },
    {
        "id": 7,
        "title": "Backend Engineer",
        "company": "ServerSide Inc",
        "location": "Chicago, IL",
        "required_skills": ["Java", "Spring", "SQL", "Docker", "Git"],
        "required_experience": 2,
        "required_education": "Bachelor",
        "salary_range": "$85,000 - $115,000",
        "description": "Design and implement robust backend services and APIs."
    },
    {
        "id": 8,
        "title": "Cloud Solutions Architect",
        "company": "NimbusTech",
        "location": "Remote",
        "required_skills": ["AWS", "Azure", "Terraform", "Kubernetes", "Docker", "Python"],
        "required_experience": 5,
        "required_education": "Bachelor",
        "salary_range": "$140,000 - $180,000",
        "description": "Design and oversee cloud architecture for enterprise clients."
    },
    {
        "id": 9,
        "title": "Data Engineer",
        "company": "PipelineData Co",
        "location": "Boston, MA",
        "required_skills": ["Python", "SQL", "AWS", "Pandas", "Docker"],
        "required_experience": 2,
        "required_education": "Bachelor",
        "salary_range": "$95,000 - $130,000",
        "description": "Build and maintain data pipelines and warehousing solutions."
    },
    {
        "id": 10,
        "title": "AI Research Scientist",
        "company": "DeepMind Labs",
        "location": "San Francisco, CA",
        "required_skills": ["Python", "PyTorch", "TensorFlow", "Keras", "NumPy"],
        "required_experience": 4,
        "required_education": "PhD",
        "salary_range": "$160,000 - $220,000",
        "description": "Research and develop cutting-edge AI/ML models."
    },
]


def load_jobs(csv_path=None):
    """
    Load jobs from CSV if available, otherwise return default jobs.
    
    CSV columns expected:
        id, title, company, location, required_skills (semicolon-separated),
        required_experience, required_education, salary_range, description
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "jobs.csv")

    if not os.path.exists(csv_path):
        return _default_jobs

    jobs = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                job = {
                    "id": int(row.get("id", 0)),
                    "title": row.get("title", "").strip(),
                    "company": row.get("company", "").strip(),
                    "location": row.get("location", "").strip(),
                    "required_skills": [s.strip() for s in row.get("required_skills", "").split(";") if s.strip()],
                    "required_experience": int(row.get("required_experience", 0)),
                    "required_education": row.get("required_education", "Bachelor").strip(),
                    "salary_range": row.get("salary_range", "N/A").strip(),
                    "description": row.get("description", "").strip(),
                }
                jobs.append(job)
        return jobs if jobs else _default_jobs
    except Exception:
        return _default_jobs
