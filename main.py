from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import pdfplumber

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "resumes"


@app.get("/")
def home():
    return {"message": "Backend Running Successfully"}


def extract_pdf_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


def calculate_score(job_description, resume_text):

    documents = [job_description, resume_text]

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])

    score = round(similarity[0][0] * 100, 2)

    return score


def skill_match(job_description, resume_text):

    skills = [
        "python",
        "sql",
        "excel",
        "power bi",
        "tableau",
        "react",
        "javascript",
        "fastapi",
        "machine learning"
    ]

    matched_skills = []

    missing_skills = []

    resume_lower = resume_text.lower()

    jd_lower = job_description.lower()

    for skill in skills:

        if skill in jd_lower:

            if skill in resume_lower:
                matched_skills.append(skill)

            else:
                missing_skills.append(skill)

    return matched_skills, missing_skills


@app.post("/upload")
async def upload_resume(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...)
):

    results = []

    for file in files:

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        resume_text = extract_pdf_text(file_path)

        score = calculate_score(
            job_description,
            resume_text
        )

        matched_skills, missing_skills = skill_match(
            job_description,
            resume_text
        )

        results.append({
            "candidate": file.filename,
            "score": score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results