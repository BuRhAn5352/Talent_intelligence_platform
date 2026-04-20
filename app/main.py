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

BASE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))

# Set tesseract path to render
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(file_bytes, filename):
    filename = filename.lower()

    if filename.endswith('.pdf'):
        # First try pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ' '.join(page.extract_text() or '' for page in pdf.pages)

        # If no text found, use PyMuPDF + OCR
        if len(text.strip()) < 50:
            print("Falling back to OCR...")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                # OCR the image
                from PIL import Image
                import io as _io
                img = Image.open(_io.BytesIO(img_bytes))
                text += pytesseract.image_to_string(img) + " "
            
            print(f"OCR got {len(text)} chars")

        return text.strip()

    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_bytes))
        return ' '.join(para.text for para in doc.paragraphs)

    return ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/hr", response_class=HTMLResponse)
async def hr_page(request: Request):
    return templates.TemplateResponse(request, "hr.html")

@app.get("/candidate", response_class=HTMLResponse)
async def candidate_page(request: Request):
    return templates.TemplateResponse(request, "candidate.html")

@app.post("/hr/analyze", response_class=HTMLResponse)
async def hr_analyze(request: Request, files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        file_bytes = await file.read()
        try:
            text = extract_text(file_bytes, file.filename)
        except Exception as e:
            print(f"Error reading {file.filename}:", e)
            text = ""

        if text and len(text.strip()) >= 20:
            result = analyze_job(text)
            result['filename'] = file.filename
            results.append(result)

    if not results:
        return templates.TemplateResponse(request, "hr.html", {
            "error": "Could not read any uploaded files."
        })

    # Group by cluster name, sorted by match score within each group
    grouped = defaultdict(list)
    for r in results:
        grouped[r['cluster_name']].append(r)
    
    # Sort within each cluster by confidence descending
    for cluster in grouped:
        grouped[cluster].sort(key=lambda x: x['confidence'], reverse=True)
    
    # Convert to sorted list of (cluster_name, [results]) by group size descending
    grouped_results = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    return templates.TemplateResponse(request, "hr_result.html", {
        "grouped_results": grouped_results,
        "total": len(results)
    })

@app.post("/candidate/analyze", response_class=HTMLResponse)
async def candidate_analyze(
    request: Request,
    file: UploadFile = File(...),
    min_exp: int = Form(default=0),
    max_exp: int = Form(default=5),
    skill_category: str = Form(default="application programming")
):
    file_bytes = await file.read()
    try:
        text = extract_text(file_bytes, file.filename)
    except Exception as e:
        print("Extraction error:", e)
        text = ""

    if not text or len(text.strip()) < 20:
        return templates.TemplateResponse(request, "candidate.html", {
            "error": "Invalid or unreadable file. Please upload a proper PDF or DOCX."
        })

    result = analyze_resume(text, min_exp, max_exp, skill_category)
    return templates.TemplateResponse(request, "candidate_result.html", {"result": result})
