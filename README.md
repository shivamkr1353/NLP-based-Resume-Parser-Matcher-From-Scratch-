# NLP‑based Resume Parser & Matcher

> Rank resumes against a job description using **TF‑IDF** and **Cosine Similarity** — every core algorithm written from scratch in Python.

🔗 **Live Demo:** [Deploy on Streamlit Cloud](https://share.streamlit.io/) *(Paste your deployed Streamlit URL here)*

---

## Overview

Recruiters paste a job description, upload resumes, and instantly see candidates ranked by relevance. The system extracts skills, highlights gaps, and explains *why* each score was assigned — all through a Streamlit dashboard.

### Core Pipeline

```
Resume → Tokenize → Remove Stop Words → Stem → TF‑IDF Vector ─┐
                                                                ├─→ Cosine Similarity → Ranked Results
Job Description → Tokenize → Remove Stop Words → Stem → TF‑IDF Vector ─┘
```

### What "From Scratch" Means

The NLP and ML algorithms below use **only** `math`, `re`, and `collections` — no scikit‑learn, no spaCy, no NLTK.

| Algorithm | Formula | Module |
|---|---|---|
| Term Frequency | `TF(t,d) = count(t in d) / total_terms(d)` | `src/vectorization/tf.py` |
| Inverse Document Frequency | `IDF(t) = log(N / (1 + df(t)))` | `src/vectorization/idf.py` |
| TF‑IDF | `TF‑IDF(t,d) = TF × IDF` | `src/vectorization/tfidf.py` |
| Cosine Similarity | `cos(A,B) = (A·B) / (‖A‖ × ‖B‖)` | `src/similarity/cosine.py` |
| Porter Stemmer | 25 suffix‑stripping rules | `src/preprocessing/stemmer.py` |
| Tokenizer | Regex‑based cleaning | `src/preprocessing/tokenizer.py` |
| Stop Words | Hand‑crafted 120+ word list | `src/preprocessing/stopwords.py` |

Utility libraries (file I/O, UI, charts) are used where they don't replace the learning objective — see `requirements.txt` for the full list.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py

# 3. Run tests (includes validation against sklearn)
python -m pytest tests/ -v
```

Check **"Use sample data for demo"** in the sidebar to try it immediately with the bundled resumes and job descriptions.

---

## Project Structure

```
├── app.py                        # Streamlit dashboard
├── requirements.txt
│
├── src/
│   ├── preprocessing/            # Tokenizer · Stop words · Stemmer
│   ├── vectorization/            # TF · IDF · TF‑IDF
│   ├── similarity/               # Cosine similarity · Ranking pipeline
│   └── parsers/                  # PDF & DOCX text extraction
│
├── data/
│   ├── sample_resumes/           # 5 sample resumes (txt)
│   ├── sample_job_descriptions/  # 3 sample JDs (txt)
│   └── skills_database.json      # 200+ skills across 8 categories
│
├── notebooks/
│   └── walkthrough.ipynb         # Step‑by‑step Jupyter walkthrough
│
└── tests/
    ├── test_preprocessing.py
    ├── test_tfidf.py
    ├── test_cosine.py
    └── test_validate_sklearn.py  # Proves our output matches sklearn
```

---

## Dashboard Features

| Tab | What It Shows |
|---|---|
| **Candidate Rankings** | Leaderboard sorted by cosine similarity, expandable skill details per candidate, CSV export |
| **Skills Analysis** | Matched vs. missing vs. bonus skills for every resume against the JD |
| **How It Works** | Live TF‑IDF vectors, dot‑product breakdown, and per‑term contribution — full math transparency |
| **Visualizations** | Score comparison bar chart, skill gap grouped bars, similarity‑vs‑skill scatter plot |

---

## Validation

`test_validate_sklearn.py` runs the same data through our from‑scratch code **and** scikit‑learn, then asserts the cosine similarity values and ranking order match. This proves correctness without relying on sklearn at runtime.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| UI | Streamlit |
| File Parsing | pdfplumber, python‑docx |
| Charts | Plotly |
| Notebook | Jupyter |
| Testing | pytest |
