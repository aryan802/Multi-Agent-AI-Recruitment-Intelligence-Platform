import streamlit as st
import pdfplumber
import re
import nltk
import os
import uuid
import chromadb
import ollama

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
# VECTOR DATABASE
# -----------------------------

chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("resume_jobs")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="resume_jobs"
)

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("AI Recruitment Intelligence Platform")

st.sidebar.info(
    """
    Semantic ATS analysis using:
    
    • Transformer embeddings\n
    • ChromaDB vector retrieval\n
    • Chunked RAG pipelines\n
    • Local LLM orchestration\n
    """
)

# -----------------------------
# MAIN TITLE
# -----------------------------

st.title("AI Recruitment Intelligence Platform")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

# -----------------------------
# SKILLS DATABASE
# -----------------------------

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

# -----------------------------
# FUNCTIONS
# -----------------------------

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


def store_documents(resume, jd):

    resume_chunks = chunk_text(resume)

    jd_chunks = chunk_text(jd)

    all_chunks = []

    metadatas = []

    ids = []

    for idx, chunk in enumerate(resume_chunks):

        all_chunks.append(chunk)

        metadatas.append({
            "type": "resume",
            "chunk": idx
        })

        ids.append(str(uuid.uuid4()))

    for idx, chunk in enumerate(jd_chunks):

        all_chunks.append(chunk)

        metadatas.append({
            "type": "job_description",
            "chunk": idx
        })

        ids.append(str(uuid.uuid4()))

    embeddings = embedding_model.encode(
        all_chunks
    ).tolist()

    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def retrieve_similar_documents(query):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results


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


def generate_rag_response(query):

    retrieval_results = retrieve_similar_documents(
        query
    )

    retrieved_docs = retrieval_results["documents"][0]

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are an expert ATS evaluator and AI recruiter.

Use ONLY the retrieved context below
to answer the question.

Retrieved Context:
{context}

Question:
{query}

Evaluate the candidate professionally.

Provide:
1. ATS compatibility assessment
2. Important missing technical skills
3. Resume improvement recommendations
4. Project recommendations for this role
5. Likely interview focus areas

Base your response ONLY on retrieved context.
"""

    try:

        response = ollama.chat(
            model='gemma:2b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    except Exception as e:

        return f"""
Local LLM unavailable.

Possible reasons:
- Ollama not installed
- gemma:2b model not downloaded
- Ollama server not running

Error:
{str(e)}
"""


# -----------------------------
# MAIN ANALYSIS
# -----------------------------

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(
        uploaded_file
    )

    cleaned_resume = clean_text(
        resume_text
    )

    cleaned_jd = clean_text(
        job_description
    )

    store_documents(
        cleaned_resume,
        cleaned_jd
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

    st.caption(
        "Score generated using semantic embeddings and skill matching."
    )

    # -----------------------------
    # SKILL ANALYSIS
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

    # -----------------------------
    # MATCHED SKILLS
    # -----------------------------

    with col1:

        st.subheader("Matched Skills")

        if matched_skills:

            for skill in matched_skills:

                st.markdown(
                    f"""
<div style="
padding:10px;
border-radius:10px;
background-color:#1e5631;
margin-bottom:10px;
color:white;
font-weight:bold;
">
✅ {skill}
</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.write(
                "No matching skills found"
            )

    # -----------------------------
    # MISSING SKILLS
    # -----------------------------

    with col2:

        st.subheader("Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.markdown(
                    f"""
<div style="
padding:10px;
border-radius:10px;
background-color:#7a1c1c;
margin-bottom:10px;
color:white;
font-weight:bold;
">
❌ {skill}
</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.write(
                "No major skills missing"
            )

    # -----------------------------
    # RESUME PREVIEW
    # -----------------------------

    st.subheader("Resume Preview")

    st.text_area(
        "",
        resume_text,
        height=300
    )

    # -----------------------------
    # RAG RESPONSE
    # -----------------------------

    st.subheader(
        "RAG Based AI Analysis"
    )

    st.caption(
    "Powered by transformer embeddings, ChromaDB vector retrieval, and local LLM orchestration."
)

    with st.spinner(
        "Generating grounded AI response..."
    ):

        feedback = generate_rag_response(
            "Analyze this resume against the job description"
        )

    st.write(feedback)

    # -----------------------------
    # RETRIEVAL RESULTS
    # -----------------------------
    show_chunks = st.sidebar.checkbox(
        "Show Retrieval Chunks"
    )

    if show_chunks:

        st.subheader(
            "Semantic Retrieval Results"
        )

        retrieval_results = retrieve_similar_documents(
            cleaned_resume
        )

        retrieved_docs = retrieval_results["documents"][0]

        for idx, doc in enumerate(retrieved_docs):

            metadata = retrieval_results["metadatas"][0][idx]

            st.markdown(
                f"""
    ### Retrieved Chunk {idx+1}

    Type: {metadata['type']}

    Chunk Index: {metadata['chunk']}
    """
            )

            st.info(doc[:500])