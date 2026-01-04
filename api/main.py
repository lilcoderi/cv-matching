import io
import re
import numpy as np
import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Menggunakan Optimum untuk menjalankan model tanpa library Torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

app = FastAPI()

# Izinkan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# KUNCI PERBAIKAN: Gunakan ONNX Runtime agar hemat RAM
# Model akan ditarik dari Hugging Face
MODEL_NAME = "lilcoderi/cv-matcher-fine-tuned"

# Memuat Tokenizer dan Model format ONNX
# export=True akan mengonversi model secara otomatis saat pertama kali dimuat
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)

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

def get_embeddings(text: str):
    """Menghasilkan vektor embedding menggunakan ONNX Runtime tanpa Torch."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np")
    outputs = model(**inputs)
    # Mean pooling: Mengambil rata-rata dari hidden states
    embeddings = outputs.last_hidden_state.mean(axis=1)
    # Normalisasi vektor agar perhitungan Cosine Similarity akurat
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm

@app.post("/match")
async def match_cvs(
    job_file: UploadFile = File(...),
    cv_files: list[UploadFile] = File(...)
):
    # 1. Proses Job Description
    job_raw = extract_text_from_pdf(await job_file.read(), max_pages=5)
    job_cleaned = clean_job_description(job_raw)
    job_final = standardize_education(clean_text(job_cleaned))
    job_embedding = get_embeddings(job_final)
    
    # 2. Proses CV
    results = []
    for cv in cv_files:
        content = await cv.read()
        raw_text = extract_text_from_pdf(content, max_pages=3)
        processed_text = standardize_education(clean_text(raw_text))
        cv_embedding = get_embeddings(processed_text)
        
        # Perhitungan Cosine Similarity manual menggunakan Numpy
        score_val = float(np.dot(job_embedding, cv_embedding.T))
        
        results.append({
            "filename": cv.filename,
            "score": round(score_val, 4),
            "percentage": round(score_val * 100, 2),
            "status": "Cocok" if score_val >= THRESHOLD else "Tidak Cocok"
        })

    # 4. Sortir Hasil
    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results}