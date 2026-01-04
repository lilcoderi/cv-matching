import io
import re
import os
import torch
import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Izinkan CORS agar frontend bisa memanggil backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# KUNCI PERBAIKAN: Gunakan ID Hugging Face, bukan path lokal
# Vercel akan mendownload ini ke RAM, sehingga tidak akan Out of Memory saat build
MODEL_NAME = "lilcoderi/cv-matcher-fine-tuned"
model = SentenceTransformer(MODEL_NAME)

THRESHOLD = 0.58

# Regex patterns
RE_CLEAN = re.compile(r'[•\-*●▪◦☑]')
RE_SPACES = re.compile(r'\s+')
RE_NON_ALPHA = re.compile(r'[^\w\s]')

def clean_text(text: str) -> str:
    text = text.lower()
    text = RE_CLEAN.sub(' ', text)
    text = text.encode("ascii", "ignore").decode()
    text = RE_NON_ALPHA.sub(' ', text)
    return RE_SPACES.sub(' ', text).strip()

def standardize_education(text: str) -> str:
    edu_map = {
        r'\b(sarjana|s1|strata 1|universitas|politeknik|institut)\b': 's1',
        r'\b(diploma 3|d3|ahli madya)\b': 'd3',
        r'\b(sma|smk|stm|smu|ma|sekolah menengah)\b': 'sma_smk',
    }
    for pattern, replacement in edu_map.items():
        text = re.sub(pattern, replacement, text)
    return text

def clean_job_description(text: str) -> str:
    noise_patterns = [
        r'we are hiring', r'send us your cv', r'kirim cv anda',
        r'hrdptoba@gmail\.com', r'subjek:.*', r'lowongan ini dibuka sampai.*'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def extract_text_from_pdf(file_bytes, max_pages=3):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        pages_to_read = min(len(pdf_reader.pages), max_pages)
        for i in range(pages_to_read):
            content = pdf_reader.pages[i].extract_text()
            if content:
                text += content + " "
        return text
    except Exception:
        raise HTTPException(status_code=400, detail="Gagal membaca file PDF")

@app.post("/match")
async def match_cvs(
    job_file: UploadFile = File(...),
    cv_files: list[UploadFile] = File(...)
):
    # 1. Proses Job Description
    job_raw = extract_text_from_pdf(await job_file.read(), max_pages=5)
    job_cleaned = clean_job_description(job_raw)
    job_final = standardize_education(clean_text(job_cleaned))
    
    # 2. Proses CV
    cv_texts_processed = []
    filenames = []

    for cv in cv_files:
        content = await cv.read()
        raw_text = extract_text_from_pdf(content, max_pages=3)
        processed_text = standardize_education(clean_text(raw_text))
        cv_texts_processed.append(processed_text)
        filenames.append(cv.filename)

    if not cv_texts_processed:
        raise HTTPException(status_code=400, detail="Tidak ada CV yang valid")

    # 3. Analisis dengan AI
    with torch.no_grad():
        job_embedding = model.encode(job_final, convert_to_tensor=True, normalize_embeddings=True)
        cv_embeddings = model.encode(cv_texts_processed, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(job_embedding, cv_embeddings)[0]

    # 4. Result
    results = []
    for i in range(len(filenames)):
        score_val = float(scores[i])
        results.append({
            "filename": filenames[i],
            "score": round(score_val, 4),
            "percentage": round(score_val * 100, 2),
            "status": "Cocok" if score_val >= THRESHOLD else "Tidak Cocok"
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results}