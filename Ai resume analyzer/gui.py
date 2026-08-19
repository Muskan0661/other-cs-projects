"""
gui.py
Tkinter GUI for the AI Resume Analyzer.
Run:  python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys, os

# Ensure the project folder is on the path
sys.path.insert(0, os.path.dirname(__file__))

from data import load_jobs
from resume_analyzer import build_resume_model, extract_features_from_meta
from matcher import JobMatcher
from solver import solve


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
PANEL_BG     = "#181c27"
CARD_BG      = "#1e2535"
ACCENT       = "#4f8ef7"
ACCENT2      = "#7c5cbf"
SUCCESS      = "#3ecf8e"
WARNING      = "#f7c948"
DANGER       = "#f75f5f"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#7a84a0"
BORDER       = "#2a3050"

FONT_TITLE   = ("Segoe UI", 22, "bold")
FONT_HEADING = ("Segoe UI", 13, "bold")
FONT_BODY    = ("Segoe UI", 10)
FONT_MONO    = ("Consolas", 10)
FONT_SMALL   = ("Segoe UI", 9)

SAMPLE_RESUME = """\
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

EDUCATION
Bachelor of Science in Computer Science – State University, 2019

PROJECTS
  • ResumeBot – NLP-powered resume screening tool (Python, Scikit-learn)
  • DashKit – real-time analytics dashboard (React, Node.js)
  • PipeFlow – ETL pipeline (Python, Pandas, PostgreSQL)

CERTIFICATIONS
AWS Certified Developer – Associate
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ─────────────────────────────────────────────────────────────────────────────

def score_colour(score):
    if score >= 70:
        return SUCCESS
    if score >= 40:
        return WARNING
    return DANGER


def make_progress_bar(parent, value, width=180, height=10, colour=None):
    """Draw a simple canvas progress bar."""
    if colour is None:
        colour = score_colour(value)
    canvas = tk.Canvas(parent, width=width, height=height,
                       bg=CARD_BG, bd=0, highlightthickness=0)
    canvas.create_rectangle(0, 0, width, height, fill=BORDER, outline="")
    filled = int(width * value / 100)
    if filled > 0:
        radius = height // 2
        canvas.create_rectangle(0, 0, filled, height, fill=colour, outline="")
    return canvas


class Tooltip:
    """Simple tooltip for any widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, background="#2a3050",
                       foreground=TEXT_PRIMARY, font=FONT_SMALL, padx=6, pady=4,
                       relief="flat")
        lbl.pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class ResumeAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Resume Analyzer")
        self.geometry("1200x780")
        self.minsize(900, 620)
        self.configure(bg=DARK_BG)

        self._recommendations = []
        self._resume_meta    = {}
        self._jobs           = load_jobs()

        self._build_styles()
        self._build_ui()

    # ── Styles ───────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame",       background=DARK_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Card.TFrame",  background=CARD_BG)

        style.configure("TLabel",
                        background=DARK_BG, foreground=TEXT_PRIMARY,
                        font=FONT_BODY)
        style.configure("Heading.TLabel",
                        background=DARK_BG, foreground=TEXT_PRIMARY,
                        font=FONT_HEADING)
        style.configure("Muted.TLabel",
                        background=DARK_BG, foreground=TEXT_MUTED,
                        font=FONT_SMALL)

        style.configure("Accent.TButton",
                        background=ACCENT, foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0, relief="flat", padding=(12, 6))
        style.map("Accent.TButton",
                  background=[("active", "#3a7de0"), ("pressed", "#2e6bc9")])

        style.configure("Ghost.TButton",
                        background=PANEL_BG, foreground=TEXT_MUTED,
                        font=FONT_SMALL, borderwidth=0, relief="flat",
                        padding=(8, 4))
        style.map("Ghost.TButton",
                  foreground=[("active", TEXT_PRIMARY)],
                  background=[("active", CARD_BG)])

        style.configure("TNotebook",
                        background=DARK_BG, borderwidth=0,
                        tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
                        background=PANEL_BG, foreground=TEXT_MUTED,
                        font=FONT_BODY, padding=[14, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", CARD_BG)],
                  foreground=[("selected", ACCENT)])

        style.configure("Treeview",
                        background=CARD_BG, foreground=TEXT_PRIMARY,
                        fieldbackground=CARD_BG, borderwidth=0,
                        font=FONT_BODY, rowheight=28)
        style.configure("Treeview.Heading",
                        background=PANEL_BG, foreground=TEXT_MUTED,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview",
                  background=[("selected", ACCENT2)],
                  foreground=[("selected", "#ffffff")])

        style.configure("Vertical.TScrollbar",
                        background=PANEL_BG, troughcolor=DARK_BG,
                        arrowcolor=TEXT_MUTED, borderwidth=0)

        style.configure("TProgressbar",
                        troughcolor=BORDER, background=ACCENT,
                        thickness=8)

    # ── Layout ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar
        header = tk.Frame(self, bg=PANEL_BG, height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="⬡  AI Resume Analyzer",
                 font=FONT_TITLE, bg=PANEL_BG, fg=ACCENT).pack(
                 side="left", padx=20, pady=8)
        tk.Label(header, text=f"  {len(self._jobs)} jobs loaded",
                 font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_MUTED).pack(
                 side="left", padx=0, pady=8)

        # ── Status bar
        self._status_var = tk.StringVar(value="Ready — paste your resume or load a file.")
        status_bar = tk.Frame(self, bg=PANEL_BG, height=26)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        self._status_lbl = tk.Label(status_bar, textvariable=self._status_var,
                                    font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_MUTED,
                                    anchor="w")
        self._status_lbl.pack(side="left", padx=12)

        # ── Main area (two panes)
        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=DARK_BG, sashrelief="flat", sashwidth=4,
                               opaqueresize=True)
        paned.pack(fill="both", expand=True, padx=0, pady=0)

        left_frame  = self._build_left_panel(paned)
        right_frame = self._build_right_panel(paned)

        paned.add(left_frame,  minsize=320, width=420)
        paned.add(right_frame, minsize=400)

    # ── Left panel (input) ────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG)

        # Title
        tk.Label(frame, text="Resume Input", font=FONT_HEADING,
                 bg=PANEL_BG, fg=TEXT_PRIMARY).pack(
                 anchor="w", padx=16, pady=(14, 2))
        tk.Label(frame, text="Paste your resume text or load a .txt file",
                 font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_MUTED).pack(
                 anchor="w", padx=16, pady=(0, 8))

        # Toolbar
        toolbar = tk.Frame(frame, bg=PANEL_BG)
        toolbar.pack(fill="x", padx=12, pady=(0, 6))

        ttk.Button(toolbar, text="📂  Load File", style="Ghost.TButton",
                   command=self._load_file).pack(side="left", padx=2)
        ttk.Button(toolbar, text="✏️  Sample Resume", style="Ghost.TButton",
                   command=self._load_sample).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🗑  Clear", style="Ghost.TButton",
                   command=self._clear_input).pack(side="left", padx=2)

        # Text area
        txt_frame = tk.Frame(frame, bg=BORDER, bd=1)
        txt_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self._text_input = tk.Text(
            txt_frame, wrap="word", font=FONT_MONO,
            bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=ACCENT,
            selectbackground=ACCENT2, selectforeground="#fff",
            relief="flat", padx=10, pady=10, undo=True
        )
        self._text_input.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(txt_frame, orient="vertical",
                            command=self._text_input.yview)
        vsb.pack(side="right", fill="y")
        self._text_input["yscrollcommand"] = vsb.set

        # Options row
        opt_frame = tk.Frame(frame, bg=PANEL_BG)
        opt_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(opt_frame, text="Top N results:", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_MUTED).pack(side="left")
        self._top_n_var = tk.IntVar(value=5)
        top_spin = tk.Spinbox(opt_frame, from_=1, to=10,
                              textvariable=self._top_n_var, width=4,
                              bg=CARD_BG, fg=TEXT_PRIMARY,
                              insertbackground=ACCENT, relief="flat",
                              buttonbackground=BORDER, font=FONT_BODY)
        top_spin.pack(side="left", padx=6)

        tk.Label(opt_frame, text="Min match %:", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_MUTED).pack(side="left", padx=(12, 0))
        self._threshold_var = tk.IntVar(value=0)
        thr_spin = tk.Spinbox(opt_frame, from_=0, to=90, increment=5,
                              textvariable=self._threshold_var, width=4,
                              bg=CARD_BG, fg=TEXT_PRIMARY,
                              insertbackground=ACCENT, relief="flat",
                              buttonbackground=BORDER, font=FONT_BODY)
        thr_spin.pack(side="left", padx=6)

        # Analyse button
        self._analyse_btn = ttk.Button(frame, text="⚡  Analyse Resume",
                                       style="Accent.TButton",
                                       command=self._run_analysis)
        self._analyse_btn.pack(fill="x", padx=12, pady=(4, 12), ipady=4)

        return frame

    # ── Right panel (results) ─────────────────────────────────────────────
    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK_BG)

        self._notebook = ttk.Notebook(frame)
        self._notebook.pack(fill="both", expand=True, padx=0, pady=0)

        # Tab 1 – Recommendations
        self._tab_recs  = tk.Frame(self._notebook, bg=DARK_BG)
        # Tab 2 – Skills profile
        self._tab_skills = tk.Frame(self._notebook, bg=DARK_BG)
        # Tab 3 – Resume summary
        self._tab_summary = tk.Frame(self._notebook, bg=DARK_BG)

        self._notebook.add(self._tab_recs,    text="  📋  Recommendations  ")
        self._notebook.add(self._tab_skills,  text="  🛠  Skill Profile  ")
        self._notebook.add(self._tab_summary, text="  📄  Resume Summary  ")

        self._build_recs_tab()
        self._build_skills_tab()
        self._build_summary_tab()

        return frame

    def _build_recs_tab(self):
        tab = self._tab_recs

        # Treeview columns
        cols = ("rank", "title", "company", "location",
                "match", "skills", "missing", "salary")
        self._tree = ttk.Treeview(tab, columns=cols, show="headings",
                                  selectmode="browse")
        col_cfg = [
            ("rank",    "#",              50,  "center"),
            ("title",   "Job Title",     200,  "w"),
            ("company", "Company",       150,  "w"),
            ("location","Location",      120,  "w"),
            ("match",   "Match %",        75,  "center"),
            ("skills",  "Skill %",        70,  "center"),
            ("missing", "Missing Skills",160,  "w"),
            ("salary",  "Salary",        140,  "w"),
        ]
        for cid, header, width, anchor in col_cfg:
            self._tree.heading(cid, text=header,
                               command=lambda c=cid: self._sort_tree(c))
            self._tree.column(cid, width=width, anchor=anchor, minwidth=40)

        self._tree.tag_configure("high",   background="#1a2e20", foreground=SUCCESS)
        self._tree.tag_configure("medium", background="#2b2515", foreground=WARNING)
        self._tree.tag_configure("low",    background="#2b1515", foreground=DANGER)
        self._tree.tag_configure("odd",    background=CARD_BG)
        self._tree.tag_configure("even",   background=PANEL_BG)

        vsb = ttk.Scrollbar(tab, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(tab, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Detail panel below
        detail_frame = tk.Frame(tab, bg=PANEL_BG, height=160)
        detail_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        detail_frame.grid_propagate(False)

        self._detail_title = tk.Label(detail_frame, text="Select a job to see details",
                                      font=FONT_HEADING, bg=PANEL_BG, fg=TEXT_MUTED,
                                      anchor="w")
        self._detail_title.pack(anchor="w", padx=14, pady=(10, 2))

        self._detail_body = tk.Text(detail_frame, font=FONT_SMALL, bg=PANEL_BG,
                                    fg=TEXT_PRIMARY, relief="flat", wrap="word",
                                    height=5, state="disabled",
                                    padx=12, pady=4)
        self._detail_body.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_skills_tab(self):
        tab = self._tab_skills
        self._skills_canvas_frame = tk.Frame(tab, bg=DARK_BG)
        self._skills_canvas_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(self._skills_canvas_frame, bg=DARK_BG,
                           bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self._skills_canvas_frame, orient="vertical",
                             command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._skills_inner = tk.Frame(canvas, bg=DARK_BG)
        self._skills_win_id = canvas.create_window(
            (0, 0), window=self._skills_inner, anchor="nw")
        self._skills_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(self._skills_win_id, width=e.width))
        self._skills_canvas = canvas

    def _build_summary_tab(self):
        tab = self._tab_summary

        frm = tk.Frame(tab, bg=DARK_BG)
        frm.pack(fill="both", expand=True)

        self._summary_text = tk.Text(frm, font=FONT_MONO, bg=DARK_BG,
                                     fg=TEXT_PRIMARY, relief="flat",
                                     wrap="word", state="disabled",
                                     padx=20, pady=16)
        vsb = ttk.Scrollbar(frm, orient="vertical",
                             command=self._summary_text.yview)
        self._summary_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._summary_text.pack(side="left", fill="both", expand=True)

        # tags for coloured output
        self._summary_text.tag_configure("header",  foreground=ACCENT,   font=("Consolas", 11, "bold"))
        self._summary_text.tag_configure("key",     foreground=SUCCESS,   font=FONT_MONO)
        self._summary_text.tag_configure("value",   foreground=TEXT_PRIMARY)
        self._summary_text.tag_configure("muted",   foreground=TEXT_MUTED)
        self._summary_text.tag_configure("warn",    foreground=WARNING)

    # ── Actions ───────────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Open resume file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self._text_input.delete("1.0", "end")
            self._text_input.insert("1.0", content)
            self._status("Loaded: " + os.path.basename(path))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _load_sample(self):
        self._text_input.delete("1.0", "end")
        self._text_input.insert("1.0", SAMPLE_RESUME)
        self._status("Sample resume loaded.")

    def _clear_input(self):
        self._text_input.delete("1.0", "end")
        self._status("Input cleared.")

    def _run_analysis(self):
        text = self._text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No Input", "Please paste a resume first.")
            return

        self._analyse_btn.config(state="disabled")
        self._status("⏳  Analysing …")
        self.update_idletasks()

        def worker():
            try:
                _, _, resume_meta = build_resume_model(text)
                resume_features   = extract_features_from_meta(resume_meta)
                matcher           = JobMatcher(resume_features)
                recs, nodes       = solve(
                    self._jobs, matcher, resume_features,
                    threshold=self._threshold_var.get(),
                    top_n=self._top_n_var.get()
                )
                self._recommendations = recs
                self._resume_meta     = resume_meta
                self.after(0, self._display_results, recs, resume_meta, nodes)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Analysis Error", str(exc)))
            finally:
                self.after(0, lambda: self._analyse_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _display_results(self, recs, resume_meta, nodes):
        self._populate_tree(recs)
        self._populate_skills(resume_meta)
        self._populate_summary(resume_meta, recs, nodes)
        self._notebook.select(0)
        self._status(f"✅  Found {len(recs)} matches — CSP explored {nodes} nodes.")

    # ── Tree population ───────────────────────────────────────────────────
    def _populate_tree(self, recs):
        for item in self._tree.get_children():
            self._tree.delete(item)

        for rank, rec in enumerate(recs, start=1):
            job = rec["job"]
            mi  = rec["match_info"]
            score = mi["match_score"]

            if score >= 70:
                tag = "high"
            elif score >= 40:
                tag = "medium"
            else:
                tag = "low"

            missing_str = ", ".join(mi["missing_skills"][:4])
            if len(mi["missing_skills"]) > 4:
                missing_str += f" (+{len(mi['missing_skills'])-4})"

            self._tree.insert("", "end", iid=str(rank-1), tags=(tag,),
                values=(
                    rank,
                    job["title"],
                    job["company"],
                    job["location"],
                    f"{score:.1f}%",
                    f"{mi['skill_match_percentage']:.0f}%",
                    missing_str or "—",
                    job.get("salary_range", "N/A"),
                ))

    def _on_tree_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._recommendations):
            return
        rec = self._recommendations[idx]
        job = rec["job"]
        mi  = rec["match_info"]

        self._detail_title.config(
            text=f"{job['title']}  ·  {job['company']}  ·  {job['location']}",
            fg=TEXT_PRIMARY
        )

        body = (
            f"Salary  : {job.get('salary_range','N/A')}\n"
            f"Match   : {mi['match_score']:.1f}%  "
            f"(Skills {mi['skill_match_percentage']:.0f}%  "
            f"Exp {mi['experience_match']:.0f}%  "
            f"Edu {mi['education_match']:.0f}%)\n"
            f"✓ Matched: {', '.join(mi['matched_skills']) or '—'}\n"
            f"✗ Missing: {', '.join(mi['missing_skills']) or '—'}\n"
            f"\n{job.get('description','')}"
        )
        self._detail_body.config(state="normal")
        self._detail_body.delete("1.0", "end")
        self._detail_body.insert("1.0", body)
        self._detail_body.config(state="disabled")

    def _sort_tree(self, col):
        """Toggle-sort treeview by a column."""
        data = [(self._tree.set(child, col), child)
                for child in self._tree.get_children("")]
        try:
            data.sort(key=lambda t: float(t[0].strip("%")))
        except ValueError:
            data.sort()
        for idx, (_, child) in enumerate(data):
            self._tree.move(child, "", idx)

    # ── Skills tab ────────────────────────────────────────────────────────
    def _populate_skills(self, resume_meta):
        for w in self._skills_inner.winfo_children():
            w.destroy()

        from data import skill_categories

        # Header
        tk.Label(self._skills_inner, text="Skill Profile",
                 font=FONT_HEADING, bg=DARK_BG, fg=TEXT_PRIMARY).pack(
                 anchor="w", padx=20, pady=(14, 4))
        tk.Label(self._skills_inner,
                 text="Skills detected in your resume grouped by category",
                 font=FONT_SMALL, bg=DARK_BG, fg=TEXT_MUTED).pack(
                 anchor="w", padx=20, pady=(0, 14))

        raw_skills = set(resume_meta.get("raw_skills", []))

        for category, cat_skills in skill_categories.items():
            found  = [s for s in cat_skills if s in raw_skills]
            absent = [s for s in cat_skills if s not in raw_skills]
            pct    = round(len(found) / len(cat_skills) * 100) if cat_skills else 0

            card = tk.Frame(self._skills_inner, bg=CARD_BG, bd=0)
            card.pack(fill="x", padx=16, pady=6)

            # Card header
            hdr = tk.Frame(card, bg=CARD_BG)
            hdr.pack(fill="x", padx=12, pady=(10, 4))

            colour = score_colour(pct)
            tk.Label(hdr, text=category, font=FONT_HEADING,
                     bg=CARD_BG, fg=TEXT_PRIMARY).pack(side="left")
            tk.Label(hdr, text=f"{pct}%", font=("Segoe UI", 11, "bold"),
                     bg=CARD_BG, fg=colour).pack(side="right")

            bar_frame = tk.Frame(card, bg=CARD_BG)
            bar_frame.pack(fill="x", padx=12, pady=(0, 6))
            bar = make_progress_bar(bar_frame, pct, width=380, colour=colour)
            bar.pack(anchor="w")

            # Chips row
            chips = tk.Frame(card, bg=CARD_BG)
            chips.pack(fill="x", padx=12, pady=(0, 10))

            for skill in found:
                chip = tk.Label(chips, text=f"✓ {skill}",
                                font=FONT_SMALL, bg="#1a3a2a", fg=SUCCESS,
                                padx=6, pady=2, relief="flat")
                chip.pack(side="left", padx=3, pady=2)

            for skill in absent[:6]:  # show up to 6 absent
                chip = tk.Label(chips, text=skill,
                                font=FONT_SMALL, bg="#252535", fg=TEXT_MUTED,
                                padx=6, pady=2, relief="flat")
                chip.pack(side="left", padx=3, pady=2)

        # Stats
        exp_val  = resume_meta.get("experience_years", {}).get("value", 0)
        edu_val  = resume_meta.get("education_level", {}).get("value", "—")
        proj_val = resume_meta.get("projects_count", {}).get("value", 0)
        cert_val = resume_meta.get("certifications_count", {}).get("value", 0)

        stats_frame = tk.Frame(self._skills_inner, bg=DARK_BG)
        stats_frame.pack(fill="x", padx=16, pady=(10, 20))

        for label, val, icon in [
            ("Experience", f"{exp_val} yrs", "⏱"),
            ("Education",  edu_val,          "🎓"),
            ("Projects",   str(proj_val),    "📁"),
            ("Certs",      str(cert_val),    "🏅"),
        ]:
            stat = tk.Frame(stats_frame, bg=CARD_BG, width=120, height=70)
            stat.pack(side="left", padx=6, pady=4)
            stat.pack_propagate(False)
            tk.Label(stat, text=icon, font=("Segoe UI", 18),
                     bg=CARD_BG, fg=ACCENT).pack(pady=(8, 0))
            tk.Label(stat, text=val, font=("Segoe UI", 11, "bold"),
                     bg=CARD_BG, fg=TEXT_PRIMARY).pack()
            tk.Label(stat, text=label, font=FONT_SMALL,
                     bg=CARD_BG, fg=TEXT_MUTED).pack()

    # ── Summary tab ───────────────────────────────────────────────────────
    def _populate_summary(self, resume_meta, recs, nodes):
        txt = self._summary_text
        txt.config(state="normal")
        txt.delete("1.0", "end")

        raw_skills = resume_meta.get("raw_skills", [])
        exp_val    = resume_meta.get("experience_years", {}).get("value", 0)
        edu_val    = resume_meta.get("education_level", {}).get("value", "—")
        proj_val   = resume_meta.get("projects_count", {}).get("value", 0)
        cert_val   = resume_meta.get("certifications_count", {}).get("value", 0)

        def h(text):
            txt.insert("end", text + "\n", "header")
        def kv(key, val, warn=False):
            txt.insert("end", f"  {key:<24}", "key")
            txt.insert("end", str(val) + "\n", "warn" if warn else "value")
        def blank():
            txt.insert("end", "\n")
        def muted(text):
            txt.insert("end", text + "\n", "muted")

        h("━━  RESUME EXTRACTION RESULTS  ━━")
        blank()
        kv("Skills detected",    len(raw_skills))
        kv("Years experience",   exp_val,  warn=(exp_val == 0))
        kv("Education level",    edu_val)
        kv("Projects found",     proj_val)
        kv("Certifications",     cert_val)
        blank()

        h("━━  DETECTED SKILLS  ━━")
        blank()
        if raw_skills:
            for i in range(0, len(raw_skills), 5):
                chunk = raw_skills[i:i+5]
                txt.insert("end", "  " + "  ·  ".join(chunk) + "\n", "value")
        else:
            muted("  (no skills detected — try adding a SKILLS section)")
        blank()

        h("━━  TOP JOB MATCHES  ━━")
        blank()
        if recs:
            for rank, rec in enumerate(recs, 1):
                j  = rec["job"]
                mi = rec["match_info"]
                sc = mi["match_score"]
                bar = "█" * int(sc // 5) + "░" * (20 - int(sc // 5))
                kv(f"#{rank}  {j['title'][:28]}", f"{sc:.1f}%  [{bar}]",
                   warn=(sc < 40))
                txt.insert("end",
                    f"       {j['company']}  ·  {j['location']}  ·  {j.get('salary_range','')}\n",
                    "muted")
                blank()
        else:
            muted("  No matches found.")

        blank()
        h("━━  CSP SOLVER STATS  ━━")
        blank()
        kv("Jobs evaluated",  len(self._jobs))
        kv("Matches returned", len(recs))
        kv("Nodes explored",   nodes)

        txt.config(state="disabled")

    # ── Helpers ───────────────────────────────────────────────────────────
    def _status(self, msg):
        self._status_var.set(msg)
        self.update_idletasks()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = ResumeAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
