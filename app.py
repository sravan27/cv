import os
import json
import shutil
import streamlit as st
from datetime import datetime
import subprocess

# Comment out repeated downloads - do once and then comment out or add a check.
import nltk
# Download only if not already available - remove repeated calls
# nltk.download('punkt_tab')

from openai import AzureOpenAI

# Print pdflatex version to confirm it's available
st.write("pdflatex version check:")
st.write(subprocess.getoutput("pdflatex --version"))

# --- Utility Functions and Definitions ---
def calculate_ats_score(text1, text2):
    from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity
    score = (
        jaccard_similarity(text1, text2) * 0.4
        + overlap_coefficient(text1, text2) * 0.3
        + cosine_similarity(text1, text2) * 0.3
    )
    return round(score * 100, 2)

def extract_job_details(job_description, model):
    job_details = model.job_details_extraction(
        job_site_content=job_description, is_st=True
    )[0]
    company = job_details.get("company_name", "Company")
    position = job_details.get("job_title", "Position")
    return job_details, company, position

def extract_user_name(user_data):
    return user_data.get("name", "Candidate")

def generate_filename(name, company, position, doc_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = name.replace(" ", "_")
    safe_company = company.replace(" ", "_")
    safe_position = position.replace(" ", "_")
    return f"{safe_name}_{safe_company}_{safe_position}_{doc_type}_{timestamp}.pdf"

def remove_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# --- Initialize Session State ---
if 'applications' not in st.session_state:
    st.session_state.applications = []

if 'generated_resume' not in st.session_state:
    st.session_state.generated_resume = None

if 'generated_cover_letter' not in st.session_state:
    st.session_state.generated_cover_letter = None

if 'interview_questions' not in st.session_state:
    st.session_state.interview_questions = ""

if 'job_description' not in st.session_state:
    st.session_state.job_description = ""

if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False
if 'coaching_mode' not in st.session_state:
    st.session_state.coaching_mode = True

st.set_page_config(
    page_title="AI Resume, Cover Letter Generator & Interview Prep",
    page_icon="📑",
    layout="wide",
)

client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-03-15-preview",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"]
)

AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

def interview_system_prompt():
    return """You are a helpful interviewer and career coach. 
You know the job description and the company's needs. 
You will:
1. Ask the candidate one interview question at a time.
2. After the candidate responds, provide constructive feedback and coaching tips on their answer.
3. Then ask the next question.
Keep the conversation professional, helpful, and structured.
"""

def generate_interview_turn(user_message, job_description):
    messages = [{"role": "system", "content": interview_system_prompt()}]
    if job_description:
        messages.append({"role": "system", "content": f"Job Description:\n{job_description}"})
    messages.extend(st.session_state.interview_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Resume & Cover Letter Generator", "Interview Preparation"]
)

if app_mode == "Resume & Cover Letter Generator":
    # Import here to avoid concurrency issues at startup
    from zlm import AutoApplyModel
    from zlm.utils.utils import display_pdf, read_file

    remove_directory("output")

    st.header("AI Resume & Cover Letter Generator", divider="rainbow")

    job_description = st.text_area(
        "Paste job description:",
        max_chars=5500,
        height=300,
        placeholder="Paste the job description here...",
    )
    st.session_state.job_description = job_description

    resume_file = st.file_uploader(
        "Upload your current resume (PDF/JSON)", type=["pdf", "json"]
    )

    col1, col2 = st.columns(2)
    with col1:
        generate_resume = st.button(
            "Generate Optimized Resume", type="primary", use_container_width=True
        )
    with col2:
        generate_cover_letter = st.button(
            "Generate Cover Letter", type="primary", use_container_width=True
        )

    if (generate_resume or generate_cover_letter) and job_description and resume_file:
        download_path = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(download_path, exist_ok=True)

        # Copy templates BEFORE calling resume_builder or cover_letter_generator
        shutil.copy("templates/resume.cls", download_path)
        shutil.copy("templates/resume.tex.jinja", download_path)

        api_key = st.secrets["OPENAI_API_KEY"]

        resume_llm = AutoApplyModel(
            api_key=api_key,
            provider="GPT",
            model="gpt-4o",
            downloads_dir=download_path,
        )

        os.makedirs("uploads", exist_ok=True)
        resume_file_path = os.path.abspath(
            os.path.join("uploads", resume_file.name)
        )
        with open(resume_file_path, "wb") as f:
            f.write(resume_file.getbuffer())

        with st.spinner("Analyzing your resume and job description..."):
            user_data = resume_llm.user_data_extraction(
                resume_file_path, is_st=True
            )
            user_name = extract_user_name(user_data)

            job_details, company, position = extract_job_details(
                job_description, resume_llm
            )

            initial_score = calculate_ats_score(
                json.dumps(user_data), json.dumps(job_details)
            )

        new_resume_generated = False
        new_cover_letter_generated = False

        if generate_resume:
            with st.spinner("Generating optimized resume..."):
                resume_path, resume_details = resume_llm.resume_builder(
                    job_details, user_data, is_st=True
                )
                st.write("Resume generated at:", resume_path)
                if not os.path.exists(resume_path):
                    st.error("Resume PDF not generated.")
                else:
                    new_score = calculate_ats_score(
                        json.dumps(resume_details), json.dumps(job_details)
                    )
                    st.session_state.generated_resume = {
                        'path': resume_path,
                        'filename': generate_filename(
                            user_name, company, position, "Resume"
                        ),
                    }
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Original ATS Score", f"{initial_score}%")
                    with c2:
                        st.metric(
                            "Optimized ATS Score",
                            f"{new_score}%",
                            delta=f"{new_score - initial_score}%",
                        )

                    st.success("✅ Resume generated successfully!")
                    new_resume_generated = True

        if generate_cover_letter:
            with st.spinner("Generating cover letter..."):
                cover_letter_details, cover_letter_path = resume_llm.cover_letter_generator(
                    job_details, user_data, is_st=True
                )
                st.write("Cover letter generated at:", cover_letter_path)
                if not os.path.exists(cover_letter_path):
                    st.error("Cover letter PDF not generated.")
                else:
                    st.session_state.generated_cover_letter = {
                        'path': cover_letter_path,
                        'filename': generate_filename(
                            user_name, company, position, "Cover_Letter"
                        ),
                        'details': cover_letter_details,
                    }
                    st.success("✅ Cover letter generated successfully!")
                    new_cover_letter_generated = True

        application_entry = {
            'company': company,
            'position': position,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'resume': st.session_state.generated_resume,
            'cover_letter': st.session_state.generated_cover_letter,
            'status': "Resume & Cover Letter Generated"
        }
        st.session_state.applications.append(application_entry)

        remove_directory("uploads")
        st.balloons()

    if st.session_state.generated_resume or st.session_state.generated_cover_letter:
        st.subheader("Your Generated Documents")

        if st.session_state.generated_resume:
            resume_info = st.session_state.generated_resume
            st.write(f"**Resume:** {resume_info['filename']}")
            if resume_info['path'] and os.path.exists(resume_info['path']):
                from zlm.utils.utils import read_file, display_pdf
                pdf_data = read_file(resume_info['path'], "rb")
                st.download_button(
                    "Download Resume ⬇",
                    data=pdf_data,
                    file_name=resume_info['filename'],
                    mime="application/pdf",
                )
                display_pdf(resume_info['path'], type="image")
            else:
                st.error("Resume file not found. Please try again.")

        if st.session_state.generated_cover_letter:
            cover_letter_info = st.session_state.generated_cover_letter
            st.write(f"**Cover Letter:** {cover_letter_info['filename']}")
            if cover_letter_info['path'] and os.path.exists(cover_letter_info['path']):
                from zlm.utils.utils import read_file
                cv_data = read_file(cover_letter_info['path'], "rb")
                st.download_button(
                    "Download Cover Letter ⬇",
                    data=cv_data,
                    file_name=cover_letter_info['filename'],
                    mime="application/pdf",
                )
                st.markdown(cover_letter_info['details'], unsafe_allow_html=True)
            else:
                st.error("Cover letter file not found. Please try again.")

    if st.session_state.applications:
        st.subheader("Job Application Tracker")
        stages = ["Resume & Cover Letter Generated", "Applied", "Interviewed", "Offer Received", "Hired", "Rejected"]
        for idx, app in enumerate(st.session_state.applications):
            with st.expander(f"Application {idx + 1}: {app['position']} at {app['company']} ({app['date']})"):
                app['status'] = st.selectbox(
                    "Application Status:", 
                    stages, 
                    index=stages.index(app['status']),
                    key=f"app_status_{idx}"
                )

                if app['resume'] and app['resume']['path'] and os.path.exists(app['resume']['path']):
                    resume_info = app['resume']
                    pdf_data = read_file(resume_info['path'], "rb")
                    st.download_button(
                        "Download Resume ⬇",
                        data=pdf_data,
                        file_name=resume_info['filename'],
                        mime="application/pdf",
                        key=f"resume_download_{idx}"
                    )
                else:
                    st.info("Resume file not available.")

                if app['cover_letter'] and app['cover_letter']['path'] and os.path.exists(app['cover_letter']['path']):
                    cover_letter_info = app['cover_letter']
                    cv_data = read_file(cover_letter_info['path'], "rb")
                    st.download_button(
                        "Download Cover Letter ⬇",
                        data=cv_data,
                        file_name=cover_letter_info['filename'],
                        mime="application/pdf",
                        key=f"cover_letter_download_{idx}"
                    )
                else:
                    st.info("Cover letter file not available.")

    else:
        st.info("No applications tracked yet. Generate a resume or cover letter to start tracking.")

elif app_mode == "Interview Preparation":
    # Import only when needed
    from zlm.utils.utils import read_file, display_pdf

    st.header("AI Interview Preparation", divider="rainbow")

    job_description = st.session_state.get('job_description', '')
    if not job_description:
        job_description = st.text_area(
            "Paste job description:",
            max_chars=5500,
            height=300,
            placeholder="Paste the job description here...",
        )
        st.session_state.job_description = job_description

    interview_type = st.selectbox(
        "Select Interview Type",
        ["General", "HR Round", "Technical Round", "Behavioral Round"]
    )

    generate_interview = st.button("Generate Interview Questions", type="primary")

    if generate_interview and job_description:
        with st.spinner("Generating interview questions..."):
            if interview_type == "General":
                system_prompt = "You are a helpful assistant generating general interview questions."
            elif interview_type == "HR Round":
                system_prompt = "You are an HR professional generating HR interview questions."
            elif interview_type == "Technical Round":
                system_prompt = "You are a technical interviewer generating technical interview questions."
            elif interview_type == "Behavioral Round":
                system_prompt = "You are a behavioral specialist generating behavioral interview questions."
            else:
                system_prompt = "You are a helpful assistant generating interview questions."

            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate 10 {interview_type} interview questions for the following job description:\n{job_description}"}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                interview_questions = response.choices[0].message.content
                st.session_state.interview_questions = interview_questions
                st.text_area("Interview Questions:", value=interview_questions, height=300)
            except Exception as e:
                st.error(f"An error occurred while generating interview questions: {e}")
    elif generate_interview and not job_description:
        st.error("Please provide the job description to generate interview questions.")
    else:
        if st.session_state.interview_questions:
            st.text_area("Interview Questions:", value=st.session_state.interview_questions, height=300)

    st.markdown("---")
    st.subheader("Interactive Interview & Coaching Session")

    st.write("""
    **Instructions:**
    - Ensure a job description is provided.
    - Click 'Start Interview' to begin an interactive Q&A session.
    - The system will ask you a question, you respond, and it will give feedback and ask the next question.
    - You can stop at any time.
    """)

    start_interview = st.button("Start Interview", disabled=st.session_state.interview_active or not job_description)
    if start_interview:
        st.session_state.interview_history = []
        st.session_state.interview_active = True
        initial_question = "Let's begin. Could you tell me how your experience aligns with this role?"
        st.session_state.interview_history.append({"role": "assistant", "content": initial_question})
        st.success("Interview session started!")
        st.write(f"**Interviewer:** {initial_question}")

    if st.session_state.interview_active:
        user_answer = st.text_area("Your answer:", height=100, placeholder="Type your response here...")
        answer_submitted = st.button("Send Answer")

        if answer_submitted and user_answer.strip():
            st.session_state.interview_history.append({"role": "user", "content": user_answer})
            response = generate_interview_turn(user_answer, st.session_state.job_description)
            st.session_state.interview_history.append({"role": "assistant", "content": response})
            st.write(f"**Interviewer/Coach:** {response}")

        stop_interview = st.button("Stop Interview")
        if stop_interview:
            st.session_state.interview_active = False
            st.success("Interview session ended.")

else:
    st.error("Invalid App Mode Selected.")

try:
    pass
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

