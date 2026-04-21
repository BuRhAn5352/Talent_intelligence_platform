# Talent_intelligence_platform

# An AI-Powered Resume & Job Analyzer

An end-to-end machine learning system that analyzes resumes and job descriptions to classify roles, estimate experience levels, detect skill gaps, and provide market-aligned salary insights.

---

##  Overview

This project builds a real-world NLP pipeline that:

* Understands resume/job text using **TF-IDF**
* Groups roles using **KMeans clustering**
* Predicts structured job categories using a **Decision Tree**
* Refines predictions using **rule-based logic**
* Uses **LDA (topic modeling)** for high-level skill gap insights
* Provides **salary benchmarks** using external APIs
* Supports **PDF/DOCX + OCR** for scanned resumes

---

## Tech Stack

* **Python**
* **FastAPI** (Backend)
* **Scikit-learn** (ML models)
* **Gensim (LDA)** (Topic modeling)
* **Pandas / NumPy**
* **Jinja2** (Frontend templating)
* **pdfplumber / python-docx / pytesseract / PyMuPDF**
* **Adzuna API** (Salary data)

---

## Features

### Resume Analysis

* Predicts **industry + role cluster**
* Computes **match score** against cluster
* Estimates **experience gap**
* Detects **secondary role (Finance, Marketing, etc.)**
* Provides **salary benchmarks**
* Extracts insights even from **scanned PDFs (OCR fallback)**

---

### HR Job Analysis

* Classifies job descriptions into **role clusters**
* Suggests **expected experience range**
* Groups candidates by **role category**

---

### Smart System Design

* ML handles **pattern recognition**
* Rules handle **precision + edge cases**
* LDA adds **topic-level understanding (optional insight)**

---

## Project Structure

```
jobanalyserproject/
│
├── app/
│   ├── main.py              # FastAPI app
│   ├── predictor.py         # ML + logic layer
│   ├── templates/           # HTML (Jinja2)
│   ├── static/              # CSS / assets
│
├── models/                  # Saved ML models
│   ├── tfidf_vectorizer.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_clf.pkl
│   ├── lda_model.pkl
│   ├── cluster_names.pkl
│   └── ...
│
├── notebooks/               # Training & experimentation
│
└── README.md
```

---

## How to Run

### 1. Clone repo

```bash
git clone https://github.com/your-username/job-analyzer.git
cd job-analyzer
```

---

### 2. Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run app

```bash
cd app
uvicorn main:app --reload
```

---

### 5. Open in browser

```
http://127.0.0.1:8000
```

---

## Model Pipeline

```
Text Input
   ↓
TF-IDF Vectorization
   ↓
Industry Classifier
   ↓
Cluster Prediction (Decision Tree)
   ↓
KMeans Match Score
   ↓
Rule-Based Refinement (Secondary Role)
   ↓
LDA Topic Analysis (Optional Insight)
   ↓
Salary Benchmark API
```

---

## Limitations

* LDA topics are **abstract**, not exact skills
* Salary API may return **incomplete data**
* Performance depends on **training dataset quality**
* Mixed-role resumes may still require **manual interpretation**

---

## Key Learnings

* ML alone is not enough → **rules improve accuracy**
* Clustering needs **clean labeling + interpretation**
* Real-world systems require **error handling + fallbacks**
* OCR is essential for handling **real resumes**

---

## Future Improvements

* Skill-level extraction (beginner/intermediate/expert)
* Better keyword weighting for refinement
* Resume ranking system for HR
* Deploy on **Render / AWS**
* Add user authentication + dashboards

---

## Author

**Burhan**
AI/ML Engineer | Building real-world systems, not just models

---

## Note

This project focuses on **practical ML system design**, not just model accuracy.

--



* demo GIF
* 2–3 screenshots
* short video
