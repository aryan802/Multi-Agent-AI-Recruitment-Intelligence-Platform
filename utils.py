import pdfplumber
import re

from sklearn.metrics.pairwise import cosine_similarity

from database import embedding_model


COMMON_SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "streamlit",
    "tensorflow",
    "pandas",
    "numpy",
    "nlp",
    "react",
    "javascript",
    "git",
    "github",
    "linux",
    "power bi",
    "tableau",
    "selenium",
    "xgboost",
    "bash",
    "flask",
    "django",
    "aws",
    "docker",
    "langchain",
    "rag",
    "llm",
    "vector database",
    "fastapi"
]


def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9 ]',
        ' ',
        text
    )

    return text


def chunk_text(
    text,
    chunk_size=400,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def extract_skills(text):

    found_skills = []

    for skill in COMMON_SKILLS:

        if skill in text:
            found_skills.append(skill)

    return found_skills


def calculate_similarity(resume, jd):

    resume_embedding = embedding_model.encode(
        [resume]
    )

    jd_embedding = embedding_model.encode(
        [jd]
    )

    semantic_similarity = cosine_similarity(
        resume_embedding,
        jd_embedding
    )[0][0]

    semantic_score = semantic_similarity * 100

    resume_skills = extract_skills(resume)

    jd_skills = extract_skills(jd)

    matched = len(
        set(resume_skills).intersection(
            set(jd_skills)
        )
    )

    total = max(len(jd_skills), 1)

    skill_score = (matched / total) * 100

    final_score = (
        0.7 * semantic_score
    ) + (
        0.3 * skill_score
    )

    return round(final_score, 2)