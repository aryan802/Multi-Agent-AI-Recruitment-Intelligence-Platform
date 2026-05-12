import ollama

from database import (
    search_resumes,
    search_jds
)


def get_context(query):

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

    return resume_context, jd_context


def run_agent(
    system_prompt,
    resume_context,
    jd_context
):

    prompt = f"""
Resume Context:
{resume_context}

Job Description Context:
{jd_context}

Task:
{system_prompt}
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

        return f"Agent Error: {str(e)}"


def ats_agent(
    resume_context,
    jd_context
):

    prompt = """
Evaluate ATS compatibility.

Analyze:
- keyword alignment
- technical skill coverage
- ATS optimization
- resume weaknesses

Do NOT say:
'context not provided'

Give concise practical feedback.
"""

    return run_agent(
        prompt,
        resume_context,
        jd_context
    )


def recruiter_agent(
    resume_context,
    jd_context
):

    prompt = """
Act as a technical recruiter.

Analyze:
- hiring suitability
- candidate strengths
- candidate weaknesses
- recruiter confidence

Do NOT say:
'context not provided'

Give concise professional evaluation.
"""

    return run_agent(
        prompt,
        resume_context,
        jd_context
    )


def interview_agent(
    resume_context,
    jd_context
):

    prompt = """
Act as a technical interviewer.

Generate:
- likely interview questions
- technical weak areas
- important concepts to prepare

Focus on practical preparation.
"""

    return run_agent(
        prompt,
        resume_context,
        jd_context
    )


def career_coach_agent(
    resume_context,
    jd_context
):

    prompt = """
Act as an AI career coach.

Suggest:
- project improvements
- missing skills
- roadmap recommendations
- portfolio improvements

Focus on career growth.
"""

    return run_agent(
        prompt,
        resume_context,
        jd_context
    )