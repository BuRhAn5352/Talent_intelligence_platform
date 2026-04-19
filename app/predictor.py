import os
import joblib
import requests
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'models')

#load all models with startup
tfidf         = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
industry_model= joblib.load(os.path.join(MODEL_DIR, 'industry_model.pkl'))
kmeans        = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
clf           = joblib.load(os.path.join(MODEL_DIR, 'cluster_clf.pkl'))
le_industry   = joblib.load(os.path.join(MODEL_DIR, 'le_industry.pkl'))
lda_model     = joblib.load(os.path.join(MODEL_DIR, 'lda_model.pkl'))
dictionary    = joblib.load(os.path.join(MODEL_DIR, 'lda_dictionary.pkl'))
le_skills     = joblib.load(os.path.join(MODEL_DIR, 'le_skills.pkl'))
cluster_names = joblib.load(os.path.join(MODEL_DIR, 'cluster_names.pkl'))

TOPIC_NAMES = {
    0: "Education & Academic Research",
    1: "Sales & Business Development",
    2: "QA & Software Development",
    3: "General Management & Leadership",
    4: "IT Infrastructure & Networking",
    5: "Web & Application Development",
    6: "Healthcare & Medical Recruitment",
    7: "BPO & Customer Service",
    8: "SAP & Enterprise Systems",
    9: "Digital Marketing & Media"
}

# Cluster experience ranges from your analysis
CLUSTER_EXP = {
    0: (2,6), 1:(4,9), 2: (1,4),
    3:(3,7), 4: (3,8), 5: (3,7),
    6: (3, 7),   7: (6, 10),  8: (5, 9),
    9: (3, 7)
}

def get_salary_benchmark(cluster_name: str):
    APP_ID = "bce508a8"
    APP_KEY = "b0b88963bc5aeb28f604d7012533c6ba"

    keyword_map = {
        "Corporate Finance, Marketing & Business Ops": "finance manager",
        "Strategic Sales & Business Development": "business development",
        "Human Resources & Talent Management": "HR manager",
        "Clinical Research & Industrial Production": "clinical research",
        "Academic Instruction & Training": "teacher trainer",
        "University Faculty & Advanced Research": "research professor",
        "Enterprise ERP Consulting & Systems": "SAP consultant",
        "Associate Operations & Service Delivery": "operations executive",
        "Mid-Senior Software Engineering": "software engineer",
        "Full-Stack & Mobile Product Development": "full stack developer",
    }
    query = keyword_map.get(cluster_name, cluster_name)

    def fetch(country_code):
        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "what": query,
            "results_per_page": 10,
            "content-type": "application/json"
        }
        return requests.get(url, params=params, timeout=5).json()

    try:
        data = fetch("in")
        results = data.get("results", [])
        currency = "₹"
        salaries = [r for r in results if r.get("salary_min") and r.get("salary_max")]

        # Fallback to GB if India has no salary data (not just no results)
        if not salaries:
            data = fetch("gb")
            results = data.get("results", [])
            currency = "£"
            salaries = [r for r in results if r.get("salary_min") and r.get("salary_max")]

        if not salaries:
            return None

        avg_min = sum(r["salary_min"] for r in salaries) / len(salaries)
        avg_max = sum(r["salary_max"] for r in salaries) / len(salaries)

        return {
            "avg_min": round(avg_min),
            "avg_max": round(avg_max),
            "midpoint": round((avg_min + avg_max) / 2),
            "currency": currency,
            "source": "India" if currency == "₹" else "UK (benchmark)",
            "sample_size": len(salaries)
        }
    except Exception as e:
        print(f"Salary API error: {e}")
        return None

 #Stopwords for LDA
STOPWORDS = {
    'bangalore','hyderabad','mumbai','delhi','chennai','location',
    'current','job','years','experience','salary','apply','candidate',
    'please','looking','pune','kolkata','gurgaon','noida','company',
    'profile','view','contact','details','download','send','like',
    'role','skills','required','good','the','and','for','with'
}

def tokenize(text):
    if not isinstance(text,str):
        return []
    return [w for w in simple_preprocess(text)
            if w not in STOPWORDS and len(w)>2]

#core predictions
def _predict_core(text):
    vec = tfidf.transform([text])

    predicted_industry = industry_model.predict(vec)[0]
    confidence = round(max(industry_model.predict_proba(vec)[0]) * 100, 1)
    try:
        bow = dictionary.doc2bow(tokenize(text))
        lda_topics = lda_model.get_document_topics(bow)
        top_topics = sorted(lda_topics, key=lambda x: x[1], reverse=True)[:3]
        top_topics = [(int(i), round(float(j), 3)) for i, j in top_topics]
    except Exception as e:
        print("LDA error:", e)
        top_topics = []
    return vec, predicted_industry, confidence, top_topics

def _get_cluster(vec, industry, skill_category, min_exp, max_exp):
    """predict cluster from structured features"""

    try:
        ind_n = le_industry.transform([industry])[0]
    except:
        ind_n = 0
    
    try:
        sk_n= le_skills.transform([skill_category])[0]
    except:
        sk_n = 0 
    
    cluster_id = clf.predict(
        pd.DataFrame(
            [[min_exp,max_exp, ind_n, sk_n]],
            columns=['Min_Exp_years','Max_Exp_years','industry_n','skills_n']
        )
    )[0]

    return int(cluster_id), cluster_names[cluster_id]

#HR function
def analyze_job(text, skill_category='application programming'):
    vec, industry, confidence, top_topics = _predict_core(text)

    #use cluster avg exp for this role type

    cluster_id, cluster_label = _get_cluster(vec, industry, skill_category,3,7)
    exp_min , exp_max = CLUSTER_EXP.get(cluster_id, (2,6))
    salary = get_salary_benchmark(cluster_label)

    return {
        'industry': industry,
        'confidence': confidence,
        'cluster_id': cluster_id,
        'cluster_name': cluster_label,
        'exp_range': f'{exp_min}–{exp_max} years',
        'top_topics': [(TOPIC_NAMES.get(i, f"Topic {i}"), round(s*100, 1)) for i, s in top_topics],
        'salary': salary
    }


def get_cluster_match_score(vec ,cluster_id):
    """
    Compare candidate TF-IDF vector against their predicted
    cluster centroid. Returns a match percentage 0-100.
    """
    centroid = kmeans.cluster_centers_[cluster_id]
    centroid = centroid.reshape(1,-1)
    score = cosine_similarity(vec, centroid)[0][0]
    return round(float(score) * 100, 1) 


def analyze_resume(text, min_exp,max_exp, skill_category):
    vec, industry, confidence , top_topics= _predict_core(text)
    cluster_id, cluster_label = _get_cluster(
        vec, industry, skill_category, min_exp, max_exp
    )
    match_score = get_cluster_match_score(vec, cluster_id)    #skill gap compare candidate exp to cluster expectatioins
    exp_min, exp_max = CLUSTER_EXP.get(cluster_id,(2 ,6))
    exp_gap = max(0,exp_min - min_exp)
    salary = get_salary_benchmark(cluster_label)

    return {
        'industry': industry,
        'confidence': confidence,
        'cluster_id': cluster_id,
        'cluster_name': cluster_label,
        'exp_range': f'{exp_min}–{exp_max} years',
        'exp_gap': exp_gap,
        'top_topics': [(TOPIC_NAMES.get(i, f"Topic {i}"), round(s*100, 1)) for i, s in top_topics],
        'match_score': match_score,
        'salary': salary 
    }

if __name__ == "__main__":
    
    sample_text = "Python developer with experience in machine learning and APIs"
    
    result = analyze_job(sample_text)

    print(result)
