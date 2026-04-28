import os
import io
import pdfplumber
import docx
import pytesseract
import fitz
import shutil
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from predictor import analyze_job, analyze_resume
from collections import defaultdict

app = FastAPI()

# Paths — always resolved relative to THIS file
BASE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static",StaticFiles(directory=os.path.join(BASE, "static")),name="static")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))


tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


#  garbled-text detector 
def is_garbled(text: str) -> bool:
    """
    Returns True when pdfplumber output looks scrambled (designed/columnar PDF).
    Heuristics: too short, abnormal average word length, or too many non-ASCII chars.
    """
    if not text or len(text.strip()) < 50:
        return True
    words = text.split()
    if not words:
        return True
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2 or avg_word_len > 20:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.15:
        return True
    return False


# PDF/DOCX text extraction 
def _ocr_pdf(file_bytes: bytes) -> str:
    doc  = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text += pytesseract.image_to_string(img) + " "
    return text.strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()

    if filename.endswith('.pdf'):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            plumber_text = ' '.join(page.extract_text() or '' for page in pdf.pages)

        if is_garbled(plumber_text):
            ocr_text = _ocr_pdf(file_bytes)
            return ocr_text if len(ocr_text) > len(plumber_text) else plumber_text.strip()

        return plumber_text.strip()

    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_bytes))
        return ' '.join(para.text for para in doc.paragraphs)

    return ""


#  Routes 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/hr", response_class=HTMLResponse)
async def hr_page(request: Request):
    return templates.TemplateResponse(request, "hr.html", {})


@app.get("/candidate", response_class=HTMLResponse)
async def candidate_page(request: Request):
    return templates.TemplateResponse(request, "candidate.html", {})

@app.post("/hr/analyze", response_class=HTMLResponse)
async def hr_analyze(request: Request, files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        file_bytes = await file.read()
        try:
            text = extract_text(file_bytes, file.filename)
        except Exception as e:
            print(f"Extraction error [{file.filename}]: {e}")
            text = ""

        if text and len(text.strip()) >= 10:
            try:
                result = analyze_job(text)
                if "error" not in result:
                    result['filename'] = file.filename
                    results.append(result)
            except Exception as e:
                print(f"Analysis failed [{file.filename}]: {e}")

    if not results:
        return templates.TemplateResponse(request,"hr.html", {
            "error":   "Could not process any of the uploaded files.",
        })

    grouped = defaultdict(list)
    for r in results:
        grouped[r['cluster_name']].append(r)

    for cluster in grouped:
        # We will sort by match_score — higher = better fit for that cluster = ranked first
        grouped[cluster].sort(key=lambda x: x.get('match_score', 0), reverse=True)

    grouped_results = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    return templates.TemplateResponse(request,"hr_result.html", {
        "grouped_results": grouped_results,
        "total":           len(results),
    })


@app.post("/candidate/analyze", response_class=HTMLResponse)
async def candidate_analyze(
    request:        Request,
    file:           UploadFile = File(...),
    min_exp:        int  = Form(default=0),
    max_exp:        int  = Form(default=5),
    skill_category: str  = Form(default="application programming"),
):
    file_bytes = await file.read()

    try:
        text = extract_text(file_bytes, file.filename)
    except Exception as e:
        print("Extraction error:", e)
        text = ""

    if not text or len(text.strip()) < 20:
        return templates.TemplateResponse(request,"candidate.html", {
            "error":   "Invalid or unreadable file. Please upload a proper PDF or DOCX.",
        })

    try:
        result = analyze_resume(text, min_exp, max_exp, skill_category)
    except Exception as e:
        print("Analysis error:", e)
        return templates.TemplateResponse(request,"candidate.html", {
            "error":   "Something went wrong during analysis.",
        })

    if isinstance(result, dict) and "error" in result:
        return templates.TemplateResponse(request,"candidate.html", {
            "error":   result["error"],
        })

    return templates.TemplateResponse(request, "candidate_result.html", {
    "result": result
    })
