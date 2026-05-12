# Multi-Agent-AI-Recruitment-Intelligence-Platform


An AI-powered recruitment and ATS intelligence platform built using semantic embeddings, vector databases, Retrieval-Augmented Generation (RAG), and local Large Language Models (LLMs).

The system performs semantic resume analysis, recruiter-focused candidate search, grounded AI feedback generation, and vector-based retrieval workflows using fully local and free AI infrastructure.

---

# Features

## Semantic ATS Matching

* Uses transformer embeddings instead of keyword-only matching
* Calculates semantic similarity between resume and job description
* Skill-based ATS scoring system

## RAG-Based AI Analysis

* Retrieval-Augmented Generation pipeline
* Retrieves relevant resume and job description chunks
* Generates grounded AI feedback using local LLMs

## Local LLM Integration

* Uses Ollama for local inference
* Supports lightweight models like `gemma:2b`
* No paid APIs required

## ChromaDB Vector Database

* Stores resume embeddings
* Stores job description embeddings
* Enables semantic retrieval and memory

## Recruiter Semantic Search

* Recruiters can search candidates using natural language queries
* Embedding-based candidate retrieval
* Semantic talent matching

## Chunked Document Retrieval

* Resume and JD documents are split into overlapping chunks
* Improves retrieval quality and contextual grounding

## Modular AI Architecture

Project separated into:

* `app.py` → Streamlit frontend
* `database.py` → Vector database logic
* `rag.py` → RAG generation pipeline
* `utils.py` → Parsing, cleaning, chunking, scoring

---

# Tech Stack

## Frontend

* Streamlit

## AI / NLP

* Sentence Transformers
* all-MiniLM-L6-v2
* Ollama
* Gemma 2B

## Vector Database

* ChromaDB

## Machine Learning

* Scikit-learn

## PDF Processing

* pdfplumber

## Language

* Python

---

# System Architecture

```text
Resume PDF + Job Description
                ↓
        Text Extraction
                ↓
         Text Cleaning
                ↓
        Overlap Chunking
                ↓
      Transformer Embeddings
                ↓
         ChromaDB Storage
                ↓
      Semantic Vector Retrieval
                ↓
      Local LLM (Ollama)
                ↓
      Grounded AI Analysis
```

---

# Project Structure

```text
resume_analyser/
│
├── app.py
├── database.py
├── rag.py
├── utils.py
│
├── requirements.txt
├── README.md
└── .env
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/aryan802/resume-analyzer.git
cd resume-analyzer
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download Ollama:

[https://ollama.com/download](https://ollama.com/download)

---

# Download Local Model

```bash
ollama run gemma:2b
```

This downloads the local LLM used for RAG analysis.

---

# Run Application

```bash
streamlit run app.py
```

---

# Current Capabilities

* Semantic ATS scoring
* Resume/JD vector retrieval
* Grounded RAG analysis
* Recruiter semantic candidate search
* Persistent vector memory
* Local AI inference
* Embedding-based matching

---

# Future Improvements

## Multi-Agent AI Workflow

Planned agents:

* ATS Agent
* Recruiter Agent
* Interview Coach Agent
* Career Roadmap Agent

## Retrieval Improvements

* Skill-aware retrieval
* Metadata ranking
* Hybrid search
* Better chunk ranking

## Production Enhancements

* FastAPI backend
* Docker deployment
* User authentication
* Resume upload history
* Dashboard analytics

## Advanced AI Features

* Interview question generation
* Resume rewriting
* Skill gap analysis
* AI evaluation metrics
* Hallucination detection

---

# Why This Project Is Different

Unlike traditional ATS checkers, this platform uses:

* semantic embeddings
* vector databases
* Retrieval-Augmented Generation
* local LLM orchestration
* AI retrieval pipelines

This makes the system significantly more intelligent and context-aware than keyword-based resume analyzers.

---

# Author

Aryan Jain

GitHub:
[https://github.com/aryan802/resume-analyzer](https://github.com/aryan802/resume-analyzer)


