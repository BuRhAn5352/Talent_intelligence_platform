# Talent Intelligence Platform

> AI-powered job market analytics for HR professionals and job seekers, built on 22,000 Indian job listings.

## What it does

Upload a resume or job description and get instant intelligence — role classification, salary benchmarks, skill gap analysis, and candidate ranking. No manual tagging. No spreadsheets. Just upload and go.

**Two dashboards. One model.**

| HR Dashboard | Candidate Dashboard |

| Upload up to 20 resumes at once | Upload your resume |
| Resumes grouped by role cluster | Role cluster + industry detection |
| Candidates ranked by fit score | Match score vs cluster centroid |
| Market salary benchmarks per role | Expected salary range |
| Skill themes required for the role | Skill gap recommendations |

---

## ML Pipeline

```
Raw Text (resume / JD)
        │
        ▼
  TF-IDF Vectorizer          ← max_features=8500, bigrams, min_df=5
   (tfidf_vectorizer.pkl)
        │
        ├──► Logistic Regression   → Industry label + confidence (97% accuracy, 10 classes)
        │     (industry_model.pkl)
        │
        ├──► KMeans (K=8)          → Cosine similarity to 8 role centroids
        │     (kmeans_model.pkl)
        │
        └──► LDA (8 topics)        → Topic distribution across resume/JD
              (lda_model.pkl)
                    │
                    ▼
          Smart cluster selection
          (LDA topic weights × affinity table)
                    │
                    ▼
            Role cluster label
            Salary · Exp band · Skill gap
```

### Models at a glance

| Model | Algorithm | Purpose | Performance |
|---|---|---|---|
| Industry Classifier | Logistic Regression | Predict industry from text | 97% accuracy |
| Role Clusterer | KMeans (K=8) | Group similar roles | Silhouette validated |
| Topic Modeller | LDA via Gensim | Skill theme extraction | 8 interpretable topics |

### Why this stack?

- **TF-IDF over embeddings** — interpretable, fast, no GPU needed. At 97% accuracy across 10 well-defined Indian industry classes, it's more than sufficient. Sentence-BERT would capture semantics better but can't explain *why* a document scores high for a term.
- **LDA over KMeans labels** — KMeans cluster IDs shift on retrain. LDA topic distribution is stable and semantically meaningful. The smart cluster selector uses LDA as the primary signal (40% weight) combined with cosine similarity (40%) and industry prior (20%).
- **Logistic Regression over tree models** — probability outputs (`predict_proba`) are used directly for confidence scoring and low-confidence fallbacks.

---

## Tech Stack

```
Backend          FastAPI + Python 3.11
Templating       Jinja2
Frontend         HTML5 + CSS3 (custom, no framework)
ML / NLP         scikit-learn · Gensim · NLTK
PDF Extraction   pdfplumber + PyMuPDF + pytesseract (OCR fallback)
Model Storage    joblib (9 .pkl artifacts, ~3.5MB total)
Salary API       Adzuna India (INR fallback estimates on 403)
Deployment       Render (free tier, Python 3.11)
```

---

## Project Structure

```
talent-intelligence-platform/
│
├── app/
│   ├── main.py              # FastAPI routes + PDF/DOCX extraction
│   ├── predictor.py         # All ML logic — prediction, clustering, salary
│   ├── templates/
│   │   ├── index.html
│   │   ├── hr.html
│   │   ├── hr_result.html
│   │   ├── candidate.html
│   │   └── candidate_result.html
│   └── static/
│       └── style.css
│
├── models/                  # 9 saved .pkl artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── industry_model.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_clf.pkl
│   ├── lda_model.pkl
│   ├── lda_dictionary.pkl
│   ├── le_industry.pkl
│   ├── le_skills.pkl
│   └── cluster_names.pkl
│
├── notebooks/
│   ├── data_cleaning.ipynb  # Dataset cleaning + EDA
│   └── modelling.ipynb      # Model training + evaluation
│
├── data/
│   └── cleaned_job_naukri.csv
│
├── render.yaml
├── requirements.txt
└── .python-version
```

---

## Dataset

- **Source**: Naukri.com job listings
- **Raw size**: ~22,000 listings
- **After cleaning**: ~16,500 usable records
- **Industries covered**: 10 (IT-Software, Banking, BPO, Healthcare, Education, Manufacturing, Recruitment, Automobile, Pharma, E-commerce)
- **Year**: 2016 — current market roles (ML Engineer, DevOps) may have lower representation

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/talent-intelligence-platform.git
cd talent-intelligence-platform

# 2. Create virtual environment
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
cd app
uvicorn main:app --reload

# 5. Open in browser
# http://127.0.0.1:8000
```

> **Tesseract OCR** (optional) — install separately for image-based PDF support.
> Windows: https://github.com/UB-Mannheim/tesseract/wiki
> Linux: `sudo apt install tesseract-ocr`
> The app works without it — OCR just won't fire for image PDFs.

---

## Features

### Smart Clustering
Role labels are derived from LDA topic distributions, not raw KMeans IDs. This makes cluster labels stable across retrains and semantically accurate even when the underlying KMeans shuffles.

### Niche Domain Detection
The Naukri dataset has no training data for lawyers, dentists, architects, government officers, or defence personnel. A keyword-based override layer fires before the ML pipeline for these domains — ensuring correct salary ranges and experience bands for roles the model has never seen.

| Niche Domain | Trigger Keywords |
|---|---|
| Legal / Law | advocate, llb, litigation, high court, ipc... |
| Government / Civil Services | upsc, ias, tehsildar, municipality... |
| Chartered Accountant | icai, ca final, articleship, ifrs... |
| Dental / Healthcare | dentist, bds, mbbs, surgeon... |
| Architecture | architect, autocad, revit, bim... |
| Defence & Police | indian army, battalion, constable... |
| School Teaching | school teacher, lesson plan, b.ed... |
| Performing Arts | kathak, bharatanatyam, choreography... |

### PDF Extraction with OCR Fallback
```
Upload PDF
    │
    ▼
pdfplumber extraction
    │
    ▼
is_garbled() check   ← avg word length, non-ASCII ratio, length heuristics
    │
    ├── No  → use pdfplumber text
    └── Yes → PyMuPDF rasterise → Tesseract OCR → use if longer
```

### HR Candidate Ranking
When multiple resumes are uploaded, candidates within each role cluster are ranked by cosine similarity to the cluster centroid. Highest fit score = ranked first = "Best Fit" badge.

### Salary Benchmarks
- Tries Adzuna India API first (live market data)
- If 403 / no salary data → uses hardcoded INR estimates per role cluster
- Salary always renders — never a blank card

---

## Known Limitations

| Limitation | Explanation |

| 2016 training data | Roles like ML Engineer, DevOps, Product Manager are underrepresented |
| IT-Software bias | ~55% of training data is IT. Ambiguous inputs lean toward IT prediction |
| Low confidence on resumes | Industry model trained on JDs — resumes use different vocabulary |
| Adzuna host restriction | Live salary API only works from whitelisted IPs (local dev) |
| No Tesseract on Render | OCR fallback unavailable on free tier — image PDFs get empty text |
| KMeans silhouette ~0.05 | Expected for high-dimensional sparse TF-IDF. Clusters are interpretable despite low score |

---

## Viva Q&A Prep

**Q: Why TF-IDF and not word embeddings?**
TF-IDF is interpretable — you can explain exactly why a document scores high for a term. At 97% accuracy for 10 well-defined industries it's more than sufficient. Sentence-BERT would capture semantics better but requires GPU inference and is harder to explain to a panel.

**Q: Why KMeans K=8?**
Validated using the elbow method and silhouette score. K=8 produces interpretable clusters that map cleanly to recognizable Indian job market segments. Low silhouette (~0.05) is expected for sparse high-dimensional TF-IDF space — not an indicator of bad clusters.

**Q: Why LDA if you already have KMeans?**
KMeans gives one hard cluster label. LDA gives a probability distribution across topics. A Data Science Manager resume might get one KMeans cluster but LDA shows 40% management + 35% technical — richer signal for skill gap analysis and more stable for labelling than raw cluster IDs.

**Q: Why not use the Decision Tree (cluster_clf) for routing?**
Verified by exhaustive grid search across all industry × skill × experience combinations: the Decision Tree predicts Academia (cluster 3) for ~85% of inputs. It was trained on imbalanced JD-level features and doesn't generalize. Replaced with a multi-signal selector combining LDA affinity, KMeans cosine similarity, and industry prior.

**Q: Why no lemmatization?**
Topic coherence was clean and interpretable without it. The marginal improvement in topic sharpness was not worth the processing cost. Documented as a future improvement.

---

## Built by

**Burhanuddin Aliasgar Contractor**
Final Year Capstone Project — Gandhinagar University

---

*Training data: Naukri.com · Deployment: Render · Stack: FastAPI + scikit-learn + Gensim*
