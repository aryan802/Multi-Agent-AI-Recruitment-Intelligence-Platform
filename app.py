import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("AI Resume Analyzer & ATS Checker")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    return text

def calculate_similarity(resume, jd):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume, jd])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])

    return round(similarity[0][0] * 100, 2)

def missing_keywords(resume, jd):

    resume_words = set(resume.split())
    jd_words = set(jd.split())

    missing = jd_words - resume_words

    important = [
        word for word in missing
        if len(word) > 4
    ]

    return important[:15]

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    score = calculate_similarity(
        cleaned_resume,
        cleaned_jd
    )

    st.subheader(f"ATS Match Score: {score}%")

    if score >= 75:
        st.success("Excellent Match")
    elif score >= 50:
        st.warning("Moderate Match")
    else:
        st.error("Low Match")

    st.subheader("Missing Keywords")

    missing = missing_keywords(
        cleaned_resume,
        cleaned_jd
    )

    if missing:
        for word in missing:
            st.write(f"- {word}")
    else:
        st.write("No major keywords missing")

    st.subheader("Resume Preview")

    st.text_area(
        "",
        resume_text,
        height=300
    )