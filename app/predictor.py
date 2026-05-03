import os
import joblib
import requests
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'models')

tfidf          = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
industry_model = joblib.load(os.path.join(MODEL_DIR, 'industry_model.pkl'))
kmeans         = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
le_industry    = joblib.load(os.path.join(MODEL_DIR, 'le_industry.pkl'))
le_skills      = joblib.load(os.path.join(MODEL_DIR, 'le_skills.pkl'))
lda_model      = joblib.load(os.path.join(MODEL_DIR, 'lda_model.pkl'))
dictionary     = joblib.load(os.path.join(MODEL_DIR, 'lda_dictionary.pkl'))
cluster_names  = joblib.load(os.path.join(MODEL_DIR, 'cluster_names.pkl'))

# LDA topic names (verified against actual pkl word distributions) 
# Topic 0: software, testing, system, technical, support
# Topic 1: sales, business, marketing, manager, retail
# Topic 2: teaching, training, academic, professor, research
# Topic 3: management, recruitment, financial, banking, project
# Topic 4: production, accounts, manufacturing, tax, audit
# Topic 5: software, design, programming, developer, web
# Topic 6: bpo, customer, call, ites, voice, service
# Topic 7: medical, engineering, pharma, healthcare, clinical
TOPIC_NAMES = {
    0: "IT Support & QA",
    1: "Sales & Marketing",
    2: "Education & Research",
    3: "Management & Finance",
    4: "Manufacturing & Accounts",
    5: "Software Development",
    6: "BPO & Customer Service",
    7: "Healthcare & Pharma",
}

# Dominant LDA topic -> semantic cluster label
# KMeans cluster IDs are NOT used for labelling (verified broken via grid search)
TOPIC_TO_LABEL = {
    0: "Software Engineering & IT Development",
    1: "Sales & Business Development",
    2: "Academia & Higher Education",
    3: "Generalist / Mixed Roles",
    4: "Industrial & Manufacturing Engineering",
    5: "Software Engineering & IT Development",
    6: "Customer Support & Operations",
    7: "Healthcare & Life Sciences",
}

# Experience bands keyed by label 
LABEL_EXP = {
    "Industrial & Manufacturing Engineering": (3,  8),
    "Sales & Business Development":           (2,  7),
    "Human Resources & Talent Management":    (1,  5),
    "Academia & Higher Education":            (5, 12),
    "Software Engineering & IT Development":  (2,  6),
    "Customer Support & Operations":          (0,  3),
    "Generalist / Mixed Roles":               (2,  6),
    "Healthcare & Life Sciences":             (3,  8),
    # Niche domains
    "Legal / Law":                            (3, 10),
    "Government / Civil Services":            (2, 15),
    "Real Estate":                            (1,  8),
    "Chartered Accountant":                   (2,  8),
    "Architecture & Design":                  (2,  7),
    "Dental / Healthcare":                    (1,  8),
    "Defence & Police":                       (2, 15),
    "Performing Arts":                        (1, 10),
    "School Teaching":                        (1,  8),
}

# Salary fallbacks (INR annual, India market 2024) 
SALARY_FALLBACK = {
    "Industrial & Manufacturing Engineering": (350000,   900000),
    "Sales & Business Development":           (300000,   900000),
    "Human Resources & Talent Management":    (280000,   700000),
    "Academia & Higher Education":            (500000,  1200000),
    "Software Engineering & IT Development":  (500000,  1800000),
    "Customer Support & Operations":          (200000,   550000),
    "Generalist / Mixed Roles":               (300000,   800000),
    "Healthcare & Life Sciences":             (350000,  1000000),
    "Legal / Law":                            (400000,  1500000),
    "Government / Civil Services":            (350000,   900000),
    "Real Estate":                            (250000,  1200000),
    "Chartered Accountant":                   (500000,  2000000),
    "Architecture & Design":                  (300000,   900000),
    "Dental / Healthcare":                    (400000,  1200000),
    "Defence & Police":                       (300000,   700000),
    "Performing Arts":                        (100000,   500000),
    "School Teaching":                        (200000,   600000),
}

SALARY_QUERY_MAP = {
    "Industrial & Manufacturing Engineering": "production engineer",
    "Sales & Business Development":           "business development manager",
    "Human Resources & Talent Management":    "HR manager",
    "Academia & Higher Education":            "assistant professor",
    "Software Engineering & IT Development":  "software engineer",
    "Customer Support & Operations":          "customer support executive",
    "Generalist / Mixed Roles":               "operations executive",
    "Healthcare & Life Sciences":             "clinical research associate",
    "Legal / Law":                            "advocate lawyer",
    "Government / Civil Services":            "government officer",
    "Real Estate":                            "real estate agent",
    "Chartered Accountant":                   "chartered accountant",
    "Architecture & Design":                  "architect",
    "Dental / Healthcare":                    "dentist",
    "Defence & Police":                       "defence officer",
    "Performing Arts":                        "performing arts",
    "School Teaching":                        "school teacher",
}

# KMeans cluster mapping for match score only 
# Verified empirically: which centroid has highest cosine sim for each domain
LABEL_TO_BEST_KMEANS = {
    "Industrial & Manufacturing Engineering": 0,
    "Sales & Business Development":           1,
    "Human Resources & Talent Management":    6,
    "Academia & Higher Education":            4,
    "Software Engineering & IT Development":  3,
    "Customer Support & Operations":          2,
    "Generalist / Mixed Roles":               6,
    "Healthcare & Life Sciences":             4,
    "Legal / Law":                            6,
    "Government / Civil Services":            6,
    "Real Estate":                            1,
    "Chartered Accountant":                   6,
    "Architecture & Design":                  0,
    "Dental / Healthcare":                    4,
    "Defence & Police":                       0,
    "Performing Arts":                        6,
    "School Teaching":                        4,
}

# Niche domains — fire BEFORE LDA 
# These have zero Naukri training data. Strong keyword match overrides ML output
# All keywords lowercase. min_req = minimum matches needed to trigger
NICHE_DOMAINS = [
    (
        "Legal / Law", "Legal / Law",
        ["advocate", "llb", "barrister", "litigation", "legal counsel",
         "high court", "supreme court", "district court", "ipc", "crpc",
         "judiciary", "solicitor", "notary", "legal advisor"],
        2,  # "advocate" alone too common — require 2
    ),
    (
        "Government / Civil Services", "Government / Civil Services",
        ["upsc", "ias ", "ips ", "ifs ", "tehsildar", "collector office",
         "municipality", "gram panchayat", "sarkari", "central government",
         "state government", "gazette", "civil servant"],
        1,
    ),
    (
        "Real Estate", "Real Estate",
        ["real estate", "property dealer", "realty", "land acquisition",
         "sqft", "sq ft", "property consultant", "real estate agent",
         "property management", "housing project", "flat sale"],
        2,
    ),
    (
        "Chartered Accountant", "Chartered Accountant (CA)",
        ["icai", "ca final", "articleship", "statutory audit", "ifrs",
         "ind as", "acca", "cpa ", "chartered accountant"],
        1,
    ),
    (
        "Architecture & Design", "Architecture & Design",
        ["architect", "autocad", "revit", "floor plan", "elevation drawing",
         "bim ", "architectural", "interior design", "urban planning",
         "structural design", "3d rendering"],
        2,
    ),
    (
        "Dental / Healthcare", "Dental / Healthcare",
        ["dentist", "bds ", "oral surgery", "dental clinic", "prosthodontics",
         "orthodontics", "endodontics", "dental surgeon", "periodontist",
         "mbbs", "md medicine", "ms surgery", "physician", "surgeon"],
        1,
    ),
    (
        "Defence & Police", "Defence & Police",
        ["indian army", "indian navy", "indian air force", "battalion",
         "regiment", "constable", "sub inspector", "police officer",
         "fir ", "defence forces", "ex-serviceman", "nda ", "cds exam"],
        1,
    ),
    (
        "School Teaching", "School Teaching",
        ["schoolteacher", "school teacher", "cbse teacher", "icse teacher",
         "primary teacher", "secondary teacher", "lesson plan",
         "classroom management", "school principal", "b.ed", "b ed ", "tet exam"],
        1,
    ),
    (
        "Performing Arts", "Performing Arts",
        ["kathak", "bharatanatyam", "classical dance", "ballet",
         "contemporary dance", "choreography", "salsa instructor",
         "jazz dance", "hip hop dance", "theatre actor",
         "stage performance", "film actor", "drama school"],
        1,
    ),
]

# NLP helpers 
STOPWORDS = {
    'bangalore', 'hyderabad', 'mumbai', 'delhi', 'chennai', 'location',
    'current', 'job', 'years', 'experience', 'salary', 'apply', 'candidate',
    'please', 'looking', 'pune', 'kolkata', 'gurgaon', 'noida', 'company',
    'profile', 'view', 'contact', 'details', 'download', 'send', 'like',
    'role', 'skills', 'required', 'good', 'the', 'and', 'for', 'with',
}


def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in simple_preprocess(text)
            if w not in STOPWORDS and len(w) > 2]


# Core functions 

def _domain_override(text):
    """Returns (label, secondary) if a niche domain matches, else (None, None)."""
    t = text.lower()
    for label, secondary, required_kws, min_req in NICHE_DOMAINS:
        if sum(1 for kw in required_kws if kw in t) >= min_req:
            return label, secondary
    return None, None


def _predict_core(text):
    """Run TF-IDF + industry model + LDA. Returns all signals."""
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    vec  = tfidf.transform([text])

    industry   = industry_model.predict(vec)[0]
    confidence = float(round(max(industry_model.predict_proba(vec)[0]) * 100, 1))

    try:
        bow           = dictionary.doc2bow(tokenize(text))
        raw_topics    = lda_model.get_document_topics(bow)
        sorted_topics = sorted(raw_topics, key=lambda x: x[1], reverse=True)
        top_topics    = [
            (TOPIC_NAMES.get(int(t), f"Topic {t}"), round(float(w) * 100, 1))
            for t, w in sorted_topics[:3]
        ]
        topic_weights  = {int(t): float(w) for t, w in raw_topics}
        dominant_topic = int(sorted_topics[0][0]) if sorted_topics else -1
    except Exception:
        top_topics     = []
        topic_weights  = {}
        dominant_topic = -1

    return vec, industry, confidence, top_topics, topic_weights, dominant_topic


def _lda_label(text, topic_weights, dominant_topic):
    """
    Derive semantic cluster label from LDA topic distribution.
    Only called when no niche domain override fires.
    Special cases handle HR, finance-only, and blended topics.
    """
    t = text.lower()

    # HR signal: topic 3 (Management/Finance) + HR-specific keywords
    if dominant_topic == 3:
        hr_kws = ['recruit', 'talent acquisition', 'hr manager', 'human resource',
                  'onboarding', 'hris', 'employee engagement', 'attrition',
                  'performance appraisal', 'manpower']
        if any(kw in t for kw in hr_kws):
            return "Human Resources & Talent Management"

    # Topic 4 = Manufacturing+Accounts mixed — split by content
    if dominant_topic == 4:
        mfg_kws = ['production', 'manufacturing', 'plant', 'maintenance',
                   'quality control', 'automobile', 'machiner', 'assembly', 'fabricat']
        fin_kws = ['chartered accountant', 'finance manager', 'audit',
                   'tally', 'ledger', 'balance sheet', 'taxation', 'bookkeeping']
        has_fin = any(kw in t for kw in fin_kws)
        has_mfg = any(kw in t for kw in mfg_kws)
        if has_fin and not has_mfg:
            return "Generalist / Mixed Roles"
        # both present or neither → use Industrial (manufacturing is more specific)

    # Topics 0 and 5 both strongly indicate software/IT
    if dominant_topic in (0, 5):
        return "Software Engineering & IT Development"

    label = TOPIC_TO_LABEL.get(dominant_topic, "Generalist / Mixed Roles")

    # If two topics are close in weight (within 60%), let second influence
    if len(topic_weights) >= 2:
        sorted_tw        = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
        top1_w, top2_w   = sorted_tw[0][1], sorted_tw[1][1]
        top2_id          = sorted_tw[1][0]
        if top2_w / (top1_w + 1e-9) > 0.60:
            label2 = TOPIC_TO_LABEL.get(top2_id, label)
            if label == "Generalist / Mixed Roles" and label2 != "Generalist / Mixed Roles":
                label = label2

    return label


def get_match_score(vec, label):
    """Cosine similarity to the KMeans centroid that best represents this label."""
    cluster_id = LABEL_TO_BEST_KMEANS.get(label, 3)
    centroid   = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    return round(float(cosine_similarity(vec, centroid)[0][0]) * 100, 1)


def get_salary_benchmark(label):
    """Try Adzuna India first. Falls back to hardcoded INR estimates on 403/failure."""
    query = SALARY_QUERY_MAP.get(label, "jobs")
    try:
        res = requests.get(
            "https://api.adzuna.com/v1/api/jobs/in/search/1",
            params={"app_id": "bce508a8", "app_key": "b0b88963bc5aeb28f604d7012533c6ba",
                    "what": query, "results_per_page": 10},
            timeout=4,
        )
        if res.status_code == 200:
            results  = res.json().get("results", [])
            salaries = [r for r in results
                        if r.get("salary_min") and r.get("salary_max")
                        and r["salary_min"] != r["salary_max"]]
            if salaries:
                lo = sum(r["salary_min"] for r in salaries) / len(salaries)
                hi = sum(r["salary_max"] for r in salaries) / len(salaries)
                return {"avg_min": int(lo), "avg_max": int(hi),
                        "midpoint": int((lo + hi) / 2), "currency": "₹",
                        "source": "Adzuna India", "sample_size": len(salaries),
                        "is_estimate": False}
    except Exception:
        pass
    lo, hi = SALARY_FALLBACK.get(label, (300000, 800000))
    return {"avg_min": lo, "avg_max": hi, "midpoint": (lo + hi) // 2,
            "currency": "₹", "source": "India market estimate",
            "sample_size": 0, "is_estimate": True}


def get_secondary_role(text):
    """
    Keyword-based role detection. Returns top 1-2 matching roles.
    Niche domains listed first (more specific keywords, higher priority).
    All keywords lowercase — text is lowercased before matching.
    Multi-word phrases preferred over single words to avoid false positives.
    """
    t = text.lower()
    role_keywords = {
        #Niche (specific — checked first) 
        "Legal / Law": [
            "advocate", "llb", "barrister", "litigation", "legal counsel",
            "high court", "ipc", "judiciary", "solicitor", "legal advisor",
        ],
        "Government / Civil Services": [
            "upsc", "ias ", "ips ", "tehsildar", "municipality",
            "gram panchayat", "civil servant", "gazette", "central government",
        ],
        "Real Estate": [
            "real estate", "property dealer", "realty", "land acquisition",
            "sqft", "property consultant", "housing project",
        ],
        "Chartered Accountant": [
            "icai", "ca final", "articleship", "statutory audit",
            "ifrs", "ind as", "acca", "chartered accountant",
        ],
        "Architecture & Design": [
            "architect", "autocad", "revit", "floor plan",
            "bim ", "architectural", "urban planning", "3d rendering",
        ],
        "Dental / Healthcare": [
            "dentist", "bds ", "oral surgery", "dental clinic",
            "mbbs", "md medicine", "ms surgery", "physician", "surgeon",
        ],
        "Defence & Police": [
            "indian army", "indian navy", "battalion", "regiment",
            "constable", "sub inspector", "ex-serviceman", "nda ",
        ],
        "School Teaching": [
            "school teacher", "cbse teacher", "lesson plan",
            "classroom management", "b.ed", "tet exam", "school principal",
        ],
        "Performing Arts": [
            "kathak", "bharatanatyam", "classical dance", "ballet",
            "choreography", "salsa instructor", "theatre actor", "drama school",
        ],
        # Core professional roles
        "Finance & Accounting": [
            "accountant", "accounts", "finance", "audit", "taxation", "gst",
            "tally", "ledger", "balance sheet", "payroll", "bookkeeping",
        ],
        "Marketing & Growth": [
            "marketing", "seo", "sem", "campaign", "branding", "social media",
            "digital marketing", "content marketing", "lead generation",
        ],
        "Analytics / Data Science": [
            "data science", "machine learning", "deep learning", "data analyst",
            "power bi", "tableau", "data visualization", "predictive analytics",
        ],
        "Human Resources": [
            "recruitment", "talent acquisition", "hr manager", "human resources",
            "onboarding", "employee engagement", "performance appraisal",
        ],
        "Sales & Business Dev": [
            "sales manager", "business development", "b2b sales",
            "key account", "revenue growth", "sales target", "deal closure",
        ],
        "Software Engineering": [
            "software engineer", "software developer", "backend developer",
            "frontend developer", "full stack", "java developer", "python developer",
            "react", "node.js", "django", "spring boot", "microservices",
        ],
        "Operations & Supply Chain": [
            "operations manager", "supply chain", "logistics", "procurement",
            "inventory management", "warehouse", "vendor management",
        ],
    }

    scores = {role: 0 for role in role_keywords}
    for role, keywords in role_keywords.items():
        for kw in keywords:
            if kw in t:
                scores[role] += 1

    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_roles[0][1] == 0:
        return "General / Unclear"
    return ", ".join(r for r, s in sorted_roles[:2] if s > 0)


# Public API 

def analyze_resume(text, min_exp, max_exp, skill_category):
    if not text or len(text.strip()) < 15:
        return {"error": "Resume too short or invalid"}

    niche_label, niche_secondary = _domain_override(text)
    vec, industry, confidence, top_topics, topic_weights, dominant_topic = _predict_core(text)

    if niche_label:
        cluster_label  = niche_label
        secondary_role = niche_secondary
    else:
        cluster_label  = _lda_label(text, topic_weights, dominant_topic)
        secondary_role = get_secondary_role(text)

    # If industry model is low confidence, surface the LDA topic name instead
    display_industry = industry if confidence >= 30 else f"{top_topics[0][0]} (estimated)" if top_topics else industry

    match_score      = get_match_score(vec, cluster_label)
    exp_min, exp_max = LABEL_EXP.get(cluster_label, (2, 6))
    exp_gap          = max(0, exp_min - min_exp)
    salary           = get_salary_benchmark(cluster_label)

    return {
        "industry":       display_industry,
        "confidence":     confidence,
        "cluster_name":   cluster_label,
        "secondary_role": secondary_role,
        "match_score":    match_score,
        "exp_range":      f"{exp_min}–{exp_max} years",
        "exp_gap":        exp_gap,
        "salary":         salary,
        "top_topics":     top_topics,
    }


def analyze_job(text, skill_category="application programming"):
    if not text or len(text.strip()) < 10:
        return {"error": "Invalid job description"}

    niche_label, niche_secondary = _domain_override(text)
    vec, industry, confidence, top_topics, topic_weights, dominant_topic = _predict_core(text)

    if niche_label:
        cluster_label  = niche_label
        secondary_role = niche_secondary
    else:
        cluster_label  = _lda_label(text, topic_weights, dominant_topic)
        secondary_role = get_secondary_role(text)

    display_industry = industry if confidence >= 30 else f"{top_topics[0][0]} (estimated)" if top_topics else industry

    exp_min, exp_max = LABEL_EXP.get(cluster_label, (2, 6))
    salary           = get_salary_benchmark(cluster_label)
    match_score      = get_match_score(vec, cluster_label)

    return {
        "industry":       display_industry,
        "confidence":     confidence,
        "cluster_name":   cluster_label,
        "secondary_role": secondary_role,
        "match_score":    match_score,
        "exp_range":      f"{exp_min}–{exp_max} years",
        "salary":         salary,
        "top_topics":     top_topics,
    }


if __name__ == "__main__":
    tests = [
        ("SOFTWARE",  "Python developer React Node.js REST APIs microservices backend software engineer Django"),
        ("SALES",     "sales manager business development b2b client revenue target pipeline crm deal closure"),
        ("FINANCE",   "chartered accountant finance audit gst tally ledger balance sheet tax payroll"),
        ("HR",        "recruitment talent acquisition hr manager human resources onboarding employee engagement"),
        ("BPO",       "customer support bpo call centre voice process associate ites inbound outbound"),
        ("ACADEMIA",  "professor teaching academic research university phd training education curriculum"),
        ("MFG",       "production manufacturing quality maintenance plant industrial engineer automobile"),
        ("HEALTH",    "clinical research pharma healthcare medical hospital doctor nurse biotech"),
        ("LAWYER",    "advocate llb high court litigation ipc legal counsel district court barrister"),
        ("CA",        "icai ca final articleship statutory audit ifrs chartered accountant taxation"),
        ("DENTAL",    "dentist bds oral surgery dental clinic prosthodontics orthodontics"),
        ("GOVT",      "upsc ias collector municipality gram panchayat central government gazette"),
        ("DANCE",     "kathak bharatanatyam classical dance choreography ballet theatre actor"),
        ("ARCHITECT", "architect autocad revit floor plan bim architectural urban planning"),
        ("DEFENCE",   "indian army battalion regiment ex-serviceman nda defence forces"),
        ("SCHOOL",    "school teacher cbse teacher lesson plan classroom management b.ed tet exam"),
    ]
    for label, text in tests:
        r = analyze_job(text)
        print(f"{label:10} -> {r['cluster_name'][:32]:32} | {r['exp_range']:12} | ₹{r['salary']['avg_min']:>8,}–{r['salary']['avg_max']:,}")
