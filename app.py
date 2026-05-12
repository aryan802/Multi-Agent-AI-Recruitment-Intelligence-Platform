import streamlit as st
import pdfplumber
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

st.title("AI Resume Analyzer and ATS Checker")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

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
    "tableau"
]

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

    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)

    return text

def calculate_similarity(resume, jd):

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2)
    )

    vectors = vectorizer.fit_transform([resume, jd])

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )

    base_score = similarity[0][0] * 100

    resume_skills = extract_skills(resume)
    jd_skills = extract_skills(jd)

    matched = len(
        set(resume_skills).intersection(set(jd_skills))
    )

    total = max(len(jd_skills), 1)

    skill_score = (matched / total) * 100

    final_score = (0.6 * base_score) + (0.4 * skill_score)

    return round(final_score, 2)

def extract_skills(text):

    found_skills = []

    for skill in COMMON_SKILLS:

        if skill in text:
            found_skills.append(skill)

    return found_skills

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    cleaned_resume = clean_text(resume_text)

    cleaned_jd = clean_text(job_description)

    score = calculate_similarity(
        cleaned_resume,
        cleaned_jd
    )

    st.subheader("ATS Match Score")

    st.progress(min(int(score), 100))

    st.metric(
        label="Match Score",
        value=f"{score}%"
    )

    if score >= 75:
        st.success("Excellent Match")
    elif score >= 50:
        st.warning("Moderate Match")
    else:
        st.error("Low Match")

    resume_skills = extract_skills(cleaned_resume)

    jd_skills = extract_skills(cleaned_jd)

    matched_skills = list(
        set(resume_skills).intersection(set(jd_skills))
    )

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Matched Skills")

        if matched_skills:
            for skill in matched_skills:
                st.write(f"✅ {skill}")
        else:
            st.write("No matching skills found")

    with col2:

        st.subheader("Missing Skills")

        if missing_skills:
            for skill in missing_skills:
                st.write(f"❌ {skill}")
        else:
            st.write("No major skills missing")


    st.subheader("Resume Preview")

    st.text_area(
        "",
        resume_text,
        height=300
    )