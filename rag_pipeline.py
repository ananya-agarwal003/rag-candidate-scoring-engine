import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

embeddings = FastEmbedEmbeddings()


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def build_vectorstore(resume_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def score_resume(resume_text: str, job_description: str) -> dict:
    vectorstore = build_vectorstore(resume_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_chunks = retriever.invoke(job_description)

    if not relevant_chunks:
        return {"result": "Score: 0/100\nReasons:\n- Resume content is not relevant to the job description."}

    context = "\n".join([f"[Chunk {i+1}]: {chunk.page_content}" for i, chunk in enumerate(relevant_chunks)])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert technical recruiter.
        Analyze the resume against the job description and provide:

        1. Overall Score: X/100
        2. Category Breakdown:
           - Skills Match: X/100
           - Experience Match: X/100
           - Education Match: X/100
        3. Reasons (3 bullet points)

        Job Description:
        {job_description}

        Relevant Resume Content:
        {context}

        Follow this exact format in your response.
        """
    )

    chain = prompt | llm
    response = chain.invoke({
        "job_description": job_description,
        "context": context
    })

    return {"result": response.content}