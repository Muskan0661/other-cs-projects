# AI Resume Analyzer

A CSP-based AI resume analyzer that matches resumes to job postings using constraint satisfaction, heuristic scoring, and backtracking search.

---

## Project Structure

```
resume_analyzer/
├── data.py              # Skill categories, education hierarchy, job loader
├── resume_analyzer.py   # CSP model builder + NLP extraction
├── matcher.py           # Weighted job-match scoring engine
├── solver.py            # Backtracking CSP solver with heuristics
├── visualize.py         # Console reports & pandas DataFrames
├── main.py              # CLI entry point
├── gui.py               # Tkinter desktop GUI
├── jobs.csv             # (optional) custom job postings
└── README.md
```

---

## Quick Start

### Install dependencies
```bash
pip install pandas tabulate
```

### Run the GUI
```bash
python gui.py
```

### Run via CLI
```bash
python main.py                    # uses built-in sample resume
python main.py my_resume.txt      # pass a .txt resume file
```

---

## How It Works

1. **Extract** — NLP regex patterns pull skills, experience, education, projects and certifications from raw resume text.
2. **Model** — A CSP model is built: variables for each skill category, experience, education, projects, certifications.
3. **Score** — `JobMatcher` computes a weighted match score (skills 50%, experience 25%, education 15%, certs 5%, projects 5%).
4. **Solve** — `solver.py` runs backtracking search with MRV-style heuristics, applying `is_consistent` constraints.
5. **Display** — Results rendered in the GUI or printed to the console.

---

## Custom Jobs (CSV)

Place a `jobs.csv` in the project folder with these columns:

| id | title | company | location | required_skills | required_experience | required_education | salary_range | description |
|----|-------|---------|----------|-----------------|---------------------|--------------------|--------------|-------------|

`required_skills` should be **semicolon-separated**, e.g. `Python;SQL;Docker`

---

## Architecture

See `model_diagram.svg` for the full system architecture diagram.

---

## Dependencies

| Package   | Purpose               |
|-----------|-----------------------|
| tkinter   | GUI (stdlib)          |
| pandas    | DataFrame reports     |
| tabulate  | Console table output  |
| re        | NLP extraction (stdlib)|

---

## Match Score Weights

| Component      | Weight |
|----------------|--------|
| Skills         | 50%    |
| Experience     | 25%    |
| Education      | 15%    |
| Certifications | 5%     |
| Projects       | 5%     |
