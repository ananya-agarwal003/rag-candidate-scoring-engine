import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Load the LLM (Groq - free and fast)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Load embedding model (runs locally, free, no API needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF file and returns its text content."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def build_vectorstore(resume_text: str):
    """Splits resume text into chunks and creates a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def score_resume(resume_text: str, job_description: str) -> dict:
    """
    Main RAG function:
    1. Builds a vector store from resume text
    2. Retrieves relevant chunks based on job description
    3. Asks the LLM to score and explain the match
    """
    vectorstore = build_vectorstore(resume_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_chunks = retriever.invoke(job_description)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert technical recruiter.
        Compare the candidate's resume content with the job description.
        Give a match score out of 100, and explain in 3-4 bullet points why.

        Job Description:
        {job_description}

        Relevant Resume Content:
        {context}

        Respond in this exact format:
        Score: <number>/100
        Reasons:
        - point 1
        - point 2
        - point 3
        """
    )

    chain = prompt | llm
    response = chain.invoke({
        "job_description": job_description,
        "context": context
    })

    return {"result": response.content}