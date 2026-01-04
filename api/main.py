import io
import re
import numpy as np
import PyPDF2
import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gunakan model ID Hugging Face Anda
MODEL_NAME = "lilcoderi/cv-matcher-fine-tuned"

# Load Tokenizer dan Session ONNX secara manual untuk menghemat RAM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Inisialisasi session tanpa menggunakan library Optimum yang berat
# Pastikan file model.onnx sudah ada di Hugging Face Anda
try:
    # Menggunakan provider CPU agar hemat memori
    session = ort.InferenceSession(
        f"https://huggingface.co/{MODEL_NAME}/resolve/main/model.onnx", 
        providers=['CPUExecutionProvider']
    )
except:
    # Fallback jika model.onnx belum tersedia secara direct link
    # Anda harus mengonversi model ke ONNX terlebih dahulu
    session = None

THRESHOLD = 0.58

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[•\-*●▪◦☑]', ' ', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def standardize_education(text: str) -> str:
    edu_map = {
        r'\b(sarjana|s1|strata 1|universitas|politeknik|institut)\b': 's1',
        r'\b(diploma 3|d3|ahli madya)\b': 'd3',
        r'\b(sma|smk|stm|smu|ma|sekolah menengah)\b': 'sma_smk',
    }
    for pattern, replacement in edu_map.items():
        text = re.sub(pattern, replacement, text)
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
    """Menghasilkan embedding menggunakan ONNX Runtime murni tanpa Torch/Optimum."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np")
    # Menyiapkan input untuk ONNX session
    onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    outputs = session.run(None, onnx_inputs)
    # Mean pooling sederhana menggunakan Numpy
    embeddings = np.mean(outputs[0], axis=1)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm

@app.post("/match")
async def match_cvs(job_file: UploadFile = File(...), cv_files: list[UploadFile] = File(...)):
    if session is None:
        raise HTTPException(status_code=500, detail="Model ONNX belum siap di server")

    job_raw = extract_text_from_pdf(await job_file.read(), max_pages=5)
    job_final = standardize_education(clean_text(job_raw))
    job_embedding = get_embeddings(job_final)
    
    results = []
    for cv in cv_files:
        content = await cv.read()
        raw_text = extract_text_from_pdf(content, max_pages=3)
        processed_text = standardize_education(clean_text(raw_text))
        cv_embedding = get_embeddings(processed_text)
        
        score_val = float(np.dot(job_embedding, cv_embedding.T))
        
        results.append({
            "filename": cv.filename,
            "score": round(score_val, 4),
            "percentage": round(score_val * 100, 2),
            "status": "Cocok" if score_val >= THRESHOLD else "Tidak Cocok"
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results}