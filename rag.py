import ollama

from database import (
    search_resumes,
    search_jds
)


def generate_rag_response(query):

    resume_results = search_resumes(query)

    jd_results = search_jds(query)

    resume_docs = resume_results["documents"][0]

    jd_docs = jd_results["documents"][0]

    resume_context = "\n\n".join(
        resume_docs
    )

    jd_context = "\n\n".join(
        jd_docs
    )

    prompt = f"""
You are an expert ATS evaluator and AI recruiter.

Analyze the candidate using the resume and job description context below.

Resume Context:
{resume_context}

Job Description Context:
{jd_context}

Instructions:
- Compare resume with job requirements
- Infer reasonable insights from the retrieved information
- Do NOT say "context not provided"
- Give concise and practical feedback

Provide:

1. ATS Match Evaluation
2. Missing Skills
3. Resume Improvements
4. Recommended Projects
5. Interview Preparation Tips
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

        return f"Local LLM Error: {str(e)}"