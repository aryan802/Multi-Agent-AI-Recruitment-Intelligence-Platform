import streamlit as st
import nltk
import os

from utils import (
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    extract_skills,
    calculate_similarity
)

from database import (
    store_resume,
    store_jd,
    search_resumes
)

from agents import (
    ats_agent,
    recruiter_agent,
    interview_agent,
    career_coach_agent
)

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Recruitment Intelligence Platform",
    layout="wide"
)

nltk.download('punkt', quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title(
    "AI Recruitment Intelligence Platform"
)

st.sidebar.info(
    """
Semantic ATS analysis using:

• Transformer embeddings
• ChromaDB vector retrieval
• Chunked RAG pipelines
• Local LLM orchestration
"""
)

show_chunks = st.sidebar.checkbox(
    "Show Retrieval Chunks"
)

# -----------------------------
# TITLE
# -----------------------------

st.title(
    "AI Recruitment Intelligence Platform"
)

st.caption(
    "Powered by transformer embeddings, ChromaDB vector retrieval, and local LLM orchestration."
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

candidate_name = st.text_input(
    "Candidate Name"
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

# -----------------------------
# MAIN APP
# -----------------------------

if uploaded_file and job_description and candidate_name:

    resume_text = extract_text_from_pdf(
        uploaded_file
    )

    cleaned_resume = clean_text(
        resume_text
    )

    cleaned_jd = clean_text(
        job_description
    )

    resume_chunks = chunk_text(
        cleaned_resume
    )

    jd_chunks = chunk_text(
        cleaned_jd
    )

    store_resume(
        resume_chunks,
        candidate_name
    )

    store_jd(
        jd_chunks,
        candidate_name
    )

    score = calculate_similarity(
        cleaned_resume,
        cleaned_jd
    )

    # -----------------------------
    # ATS SCORE
    # -----------------------------

    st.subheader("ATS Match Score")

    st.progress(
        min(int(score), 100)
    )

    st.metric(
        label="Match Score",
        value=f"{score}%"
    )

    if score >= 75:

        st.success(
            "Excellent Match"
        )

    elif score >= 50:

        st.warning(
            "Moderate Match"
        )

    else:

        st.error(
            "Low Match"
        )

    # -----------------------------
    # SKILLS
    # -----------------------------

    resume_skills = extract_skills(
        cleaned_resume
    )

    jd_skills = extract_skills(
        cleaned_jd
    )

    matched_skills = list(
        set(resume_skills).intersection(
            set(jd_skills)
        )
    )

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Matched Skills"
        )

        for skill in matched_skills:

            st.success(skill)

    with col2:

        st.subheader(
            "Missing Skills"
        )

        for skill in missing_skills:

            st.error(skill)

    # -----------------------------
    # MULTI AGENT AI ANALYSIS
    # -----------------------------
    from agents import get_context

    resume_context, jd_context = get_context(
        query
    )

    query = cleaned_resume + "\n" + cleaned_jd

    st.subheader(
        "ATS Agent"
    )

    with st.spinner(
        "Running ATS analysis..."
    ):

        ats_feedback = ats_agent(
            resume_context,
            jd_context
        )

    st.write(ats_feedback)

    # -----------------------------

    st.subheader(
        "Recruiter Agent"
    )

    with st.spinner(
        "Running recruiter evaluation..."
    ):

        recruiter_feedback = recruiter_agent(query)

    st.write(recruiter_feedback)

    # -----------------------------

    st.subheader(
        "Interview Agent"
    )

    with st.spinner(
        "Generating interview preparation..."
    ):

        interview_feedback = interview_agent(query)

    st.write(interview_feedback)

    # -----------------------------

    st.subheader(
        "Career Coach Agent"
    )

    with st.spinner(
        "Generating career roadmap..."
    ):

        coach_feedback = career_coach_agent(query)

    st.write(coach_feedback)


    # -----------------------------
    # RETRIEVAL DEBUG
    # -----------------------------

    if show_chunks:

        st.subheader(
            "Retrieved Resume Chunks"
        )

        retrieval_results = search_resumes(
            cleaned_resume
        )

        retrieved_docs = retrieval_results["documents"][0]

        for idx, doc in enumerate(retrieved_docs):

            st.info(doc[:500])

    # -----------------------------
    # RECRUITER SEARCH
    # -----------------------------

    st.subheader(
        "Recruiter Semantic Search"
    )

    recruiter_query = st.text_input(
        "Search Candidates by Skills or Role"
    )

    if recruiter_query:

        search_results = search_resumes(
            recruiter_query
        )

        retrieved_docs = search_results["documents"][0]

        metadatas = search_results["metadatas"][0]

        shown_candidates = set()

        for idx, doc in enumerate(retrieved_docs):

            candidate = metadatas[idx]["candidate"]

            if candidate not in shown_candidates:

                shown_candidates.add(candidate)

                st.success(
                    f"Matched Candidate: {candidate}"
                )

                st.caption(
                    "Semantic profile match found using embedding similarity."
                )