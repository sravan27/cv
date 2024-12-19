'''
-----------------------------------------------------------------------
File: app.py
Creation Time: Jan 30th 2024, 11:00 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (...)
-----------------------------------------------------------------------
'''

import os
import json
import base64
import shutil
import zipfile
import subprocess
import streamlit as st
import nltk

from zlm import AutoApplyModel
from zlm.utils.utils import display_pdf, read_file
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-03-15-preview",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"]
)
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

st.set_page_config(
    page_title="Resume Generator",
    page_icon="📑",
    menu_items={
        'Get help': 'https://github.com/Ztrimus/job-llm/issues',
        'About': 'https://github.com/Ztrimus/job-llm',
        'Report a bug': "https://github.com/Ztrimus/job-llm/issues",
    }
)

st.write("Checking pdflatex availability:")
pdflatex_version = subprocess.getoutput("pdflatex --version")
st.write(pdflatex_version)

# Just run playwright install if needed (no sudo)
os.system("playwright install")

def calculate_ats_score(text1, text2):
    score = (
        jaccard_similarity(text1, text2) * 0.4
        + overlap_coefficient(text1, text2) * 0.3
        + cosine_similarity(text1, text2) * 0.3
    )
    return round(score * 100, 2)

def interview_system_prompt():
    return """You are a helpful interviewer and career coach. 
You know the job description and the company's needs. 
You will:
1. Ask the candidate one interview question at a time.
2. After the candidate responds, provide constructive feedback and coaching tips.
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

def encode_tex_file(file_path):
    try:
        tex_file = file_path.replace('.pdf', '.tex')
        cls_file = os.path.join('output', 'resume.cls')
        zip_file_path = file_path.replace('.pdf', '.zip')
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            if os.path.exists(tex_file):
                zipf.write(tex_file, os.path.basename(tex_file))
            if os.path.exists(cls_file):
                zipf.write(cls_file, 'resume.cls')
        if os.path.exists(zip_file_path):
            with open(zip_file_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            encoded_zip = base64.b64encode(zip_content).decode('utf-8')
            return encoded_zip
        else:
            return None
    except Exception as e:
        st.error(f"An error occurred while encoding the file: {e}")
        return None

def create_overleaf_button(resume_path):
    tex_content = encode_tex_file(resume_path)
    if tex_content is None:
        return
    html_code = f"""
    <form action="https://www.overleaf.com/docs" method="post" target="_blank">
        <input type="text" name="snip_uri" style="display: none;"
            value="data:application/zip;base64,{tex_content}">
        <input class="btn btn-success rounded-pill w-100" type="submit" value="Edit in Overleaf 🍃">
    </form>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def extract_user_name(user_data):
    return user_data.get("name", "Candidate")

def generate_filename(name, company, position, doc_type):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = name.replace(" ", "_")
    safe_company = company.replace(" ", "_")
    safe_position = position.replace(" ", "_")
    return f"{safe_name}_{safe_company}_{safe_position}_{doc_type}_{timestamp}.pdf"

def remove_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def extract_job_details(job_description, model):
    job_details = model.job_details_extraction(job_site_content=job_description, is_st=True)[0]
    return job_details, job_details.get("company_name","Company"), job_details.get("job_title","Position")

# Initialize Session State
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

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Resume & Cover Letter Generator", "Interview Preparation"])

if app_mode == "Resume & Cover Letter Generator":
    st.header("Get :green[Job Aligned] :orange[Personalized] Resume", divider='rainbow')

    job_description = st.text_area("Paste job description:", max_chars=5500, height=300, placeholder="Paste the job description here...")
    st.session_state.job_description = job_description

    resume_file = st.file_uploader("Upload your current resume (PDF/JSON)", type=["pdf", "json"])

    col1, col2, col3 = st.columns(3)
    with col1:
        generate_resume = st.button("Get Resume", type="primary", use_container_width=True)
    with col2:
        generate_cover_letter = st.button("Get Cover Letter", type="primary", use_container_width=True)
    with col3:
        generate_both = st.button("Resume + Cover letter", type="primary", use_container_width=True)
        if generate_both:
            generate_resume = True
            generate_cover_letter = True

    if (generate_resume or generate_cover_letter) and job_description and resume_file:
        # Use Azure OpenAI + GPT-4o
        api_key = st.secrets["OPENAI_API_KEY"]
        resume_llm = AutoApplyModel(
            api_key=api_key,
            provider="GPT",
            model="gpt-4o",
            downloads_dir="output"
        )

        # Save user resume
        os.makedirs("uploads", exist_ok=True)
        resume_file_path = os.path.join("uploads", resume_file.name)
        with open(resume_file_path, "wb") as f:
            f.write(resume_file.getbuffer())

        remove_directory("output")
        os.makedirs("output", exist_ok=True)
        shutil.copy("templates/resume.cls", "output")
        shutil.copy("templates/resume.tex.jinja", "output")

        with st.spinner("Analyzing your resume and job description..."):
            user_data = resume_llm.user_data_extraction(resume_file_path, is_st=True)
            user_name = extract_user_name(user_data)
            job_details, company, position = extract_job_details(job_description, resume_llm)
            initial_score = calculate_ats_score(json.dumps(user_data), json.dumps(job_details))

        # Change to output directory so all generated files go here
        original_cwd = os.getcwd()
        os.chdir("output")

        if generate_resume:
            with st.spinner("Generating optimized resume..."):
                resume_path, resume_details = resume_llm.resume_builder(job_details, user_data, is_st=True)
            # Return to original directory for reading
            os.chdir(original_cwd)
            if not resume_path or not os.path.exists(resume_path):
                st.error("No resume PDF generated. Please check logs or try again.")
            else:
                new_score = calculate_ats_score(json.dumps(resume_details), json.dumps(job_details))
                st.session_state.generated_resume = {
                    'path': resume_path,
                    'filename': generate_filename(user_name, company, position, "Resume"),
                }
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Original ATS Score", f"{initial_score}%")
                with c2:
                    st.metric("Optimized ATS Score", f"{new_score}%", delta=f"{new_score - initial_score}%")
                st.success("✅ Resume generated successfully!")
                pdf_data = read_file(resume_path, "rb")
                st.download_button("Download Resume ⬇", data=pdf_data, file_name=st.session_state.generated_resume['filename'], mime="application/pdf")
                display_pdf(resume_path, type="image")
                create_overleaf_button(resume_path)
                os.chdir("output")

        if generate_cover_letter:
            with st.spinner("Generating cover letter..."):
                cv_details, cv_path = resume_llm.cover_letter_generator(job_details, user_data, is_st=True)
            os.chdir(original_cwd)
            if not cv_path or not os.path.exists(cv_path):
                st.error("No cover letter PDF generated. Please check logs or try again.")
            else:
                st.session_state.generated_cover_letter = {
                    'path': cv_path,
                    'filename': generate_filename(user_name, company, position, "Cover_Letter"),
                    'details': cv_details,
                }
                st.success("✅ Cover letter generated successfully!")
                cv_data = read_file(cv_path, "rb")
                st.download_button("Download Cover Letter ⬇", data=cv_data, file_name=st.session_state.generated_cover_letter['filename'], mime="application/pdf")
                st.markdown(cv_details, unsafe_allow_html=True)
                os.chdir("output")

        # Return to original directory
        os.chdir(original_cwd)

        application_entry = {
            'company': company,
            'position': position,
            'date': json.dumps(job_details.get("extraction_date", "")),
            'resume': st.session_state.generated_resume,
            'cover_letter': st.session_state.generated_cover_letter,
            'status': "Resume & Cover Letter Generated"
        }
        st.session_state.applications.append(application_entry)

        remove_directory("uploads")
        st.balloons()

    if st.session_state.applications:
        st.subheader("Job Application Tracker")
        stages = ["Resume & Cover Letter Generated", "Applied", "Interviewed", "Offer Received", "Hired", "Rejected"]
        for idx, app in enumerate(st.session_state.applications):
            with st.expander(f"Application {idx + 1}: {app['position']} at {app['company']} ({app['date']})"):
                current_status = app['status']
                app['status'] = st.selectbox("Application Status:", stages, index=stages.index(current_status), key=f"app_status_{idx}")

                if app.get('resume') and app['resume'].get('path') and os.path.exists(app['resume']['path']):
                    resume_info = app['resume']
                    pdf_data = read_file(resume_info['path'], "rb")
                    st.download_button("Download Resume ⬇", data=pdf_data, file_name=resume_info['filename'], mime="application/pdf", key=f"resume_download_{idx}")
                else:
                    st.info("Resume file not available.")

                if app.get('cover_letter') and app['cover_letter'].get('path') and os.path.exists(app['cover_letter']['path']):
                    cover_letter_info = app['cover_letter']
                    cv_data = read_file(cover_letter_info['path'], "rb")
                    st.download_button("Download Cover Letter ⬇", data=cv_data, file_name=cover_letter_info['filename'], mime="application/pdf", key=f"cover_letter_download_{idx}")
                else:
                    st.info("Cover letter file not available.")
    else:
        st.info("No applications tracked yet. Generate a resume or cover letter to start tracking.")

elif app_mode == "Interview Preparation":
    st.header("AI Interview Preparation", divider="rainbow")

    job_description = st.session_state.get('job_description', '')
    if not job_description:
        job_description = st.text_area("Paste job description:", max_chars=5500, height=300, placeholder="Paste the job description here...")
        st.session_state.job_description = job_description

    interview_type = st.selectbox("Select Interview Type", ["General", "HR Round", "Technical Round", "Behavioral Round"])
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
    - The system will ask you a question, you respond, then it will give feedback and ask the next question.
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

try:
    pass
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

st.markdown("[Report Feedback, Issues, or Contribute!](https://github.com/Ztrimus/job-llm/issues)")
