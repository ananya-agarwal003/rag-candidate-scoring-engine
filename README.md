# RAG-Powered Candidate Scoring Engine

A FastAPI-based backend that uses Retrieval-Augmented Generation (RAG) to score and rank resumes against a job description.

## Features
- Upload resume PDF and get an AI-generated match score
- Category breakdown: Skills, Experience, Education
- Multiple resume ranking against one job description
- Similarity threshold to filter irrelevant resumes

## Tech Stack
- **Backend:** FastAPI, Python
- **RAG Pipeline:** LangChain, FAISS, HuggingFace Embeddings (all-MiniLM-L6-v2)
- **LLM:** Groq (llama-3.1-8b-instant)
- **PDF Parsing:** PyPDF
- **Frontend:** HTML, CSS, JavaScript

## Setup

1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `.\venv\Scripts\Activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file and add: `GROQ_API_KEY=your_key_here`
6. Run server: `uvicorn main:app --reload`
7. Open `index.html` in browser

## API Endpoints
- `GET /` - Health check
- `POST /score` - Score single resume against JD
- `POST /rank` - Rank multiple resumes against JD