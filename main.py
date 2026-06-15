from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from rag_pipeline import extract_text_from_pdf, score_resume

app = FastAPI(title="RAG Candidate Scoring Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "RAG Candidate Scoring Engine is running"}


@app.post("/score")
async def score(resume: UploadFile, job_description: str = Form(...)):
    # Save uploaded resume temporarily
    temp_path = f"temp_{resume.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    # Extract text and run scoring
    resume_text = extract_text_from_pdf(temp_path)
    result = score_resume(resume_text, job_description)

    # Clean up temp file
    os.remove(temp_path)

    return result