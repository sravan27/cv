'''
-----------------------------------------------------------------------
File: app.py
Creation Time: Jan 30th 2024, 11:00 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024
-----------------------------------------------------------------------
'''

import os
import json
import base64
import shutil
import zipfile
import subprocess
import streamlit as st

from zlm import AutoApplyModel
from zlm.utils.utils import display_pdf, download_pdf, read_file, read_json
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity
from zlm.variables import LLM_MAPPING

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Azure OpenAI Configuration (no user choice)
from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-03-15-preview",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"]
)
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

print("Installing playwright...")
os.system("playwright install")
os.system("sudo playwright install-deps")

st.set_page_config(
    page_title="Resume Generator",
    page_icon="📑",
    menu_items={
        'Get help': 'https://www.youtube.com/watch?v=Agl7ugyu1N4',
        'About': 'https://github.com/Ztrimus/job-llm',
        'Report a bug': "https://github.com/Ztrimus/job-llm/issues",
    }
)

# Debug: Check pdflatex availability
st.write("Checking pdflatex availability:")
pdflatex_version = subprocess.getoutput("pdflatex --version")
st.write(pdflatex_version)

if os.path.exists("output"):
    shutil.rmtree("output")

def encode_tex_file(file_path):
    try:
        current_loc = os.path.dirname(__file__)
        file_paths = [file_path.replace('.pdf', '.tex'), os.path.join(current_loc, 'zlm', 'templates', 'resume.cls')]
        zip_file_path = file_path.replace('.pdf', '.zip')

        # Create a zip file
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for fp in file_paths:
                if os.path.exists(fp):
                    zipf.write(fp, os.path.basename(fp))

        # Read the zip file content as bytes
        if os.path.exists(zip_file_path):
            with open(zip_file_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            # Encode the data using Base64
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
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Overleaf Button</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body style="background: transparent;">
        <div style="max-height: 30px !important;">
            <form action="https://www.overleaf.com/docs" method="post" target="_blank" height="20px">
                <input type="text" name="snip_uri" style="display: none;"
                    value="data:application/zip;base64,{tex_content}">
                <input class="btn btn-success rounded-pill w-100" type="submit" value="Edit in Overleaf 🍃">
            </form>
        </div>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=40)

def calculate_ats_score(text1, text2):
    # As per previous code using jaccard, overlap, cosine
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

def extract_job_details(job_description, model):
    job_details = model.job_details_extraction(job_site_content=job_description, is_st=True)[0]
    company = job_details.get("company_name", "Company")
    position = job_details.get("job_title", "Position")
    return job_details, company, position

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

# Session State
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

    # Remove provider/model selection -> Hardcode provider="GPT" and model="gpt-4o"
    # Remove URL toggle
    st.write("Job Description Text")
    text = st.text_area("Paste job description text:", max_chars=5500, height=200, placeholder="Paste job description text here...")

    file = st.file_uploader("Upload your resume or any work-related data(PDF, JSON)", type=["json", "pdf"])

    colb1, colb2, colb3 = st.columns(3)
    with colb1:
        get_resume_button = st.button("Get Resume", key="get_resume", type="primary", use_container_width=True)
    with colb2:
        get_cover_letter_button = st.button("Get Cover Letter", key="get_cover_letter", type="primary", use_container_width=True)
    with colb3:
        get_both = st.button("Resume + Cover letter", key="both", type="primary", use_container_width=True)
        if get_both:
            get_resume_button = True
            get_cover_letter_button = True

    if (get_resume_button or get_cover_letter_button) and file and text:
        # Hardcode model and provider
        api_key = st.secrets["OPENAI_API_KEY"]
        resume_llm = AutoApplyModel(
            api_key=api_key,
            provider="GPT",
            model="gpt-4o",
            downloads_dir="output"
        )

        # Save the uploaded file
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.abspath(os.path.join("uploads", file.name))
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
        # Extract user data
        with st.status("Extracting user data..."):
            user_data = resume_llm.user_data_extraction(file_path, is_st=True)
            st.write(user_data)

        shutil.rmtree(os.path.dirname(file_path))

        if user_data is None:
            st.error("User data not able to process. Please upload a valid file")
            st.stop()

        # Extract job details
        with st.status("Extracting job details..."):
            job_details, jd_path = resume_llm.job_details_extraction(job_site_content=text, is_st=True)
            st.write(job_details)

        if job_details is None:
            st.error("Please paste job description text and try again!")
            st.stop()

        user_name = extract_user_name(user_data)
        company = job_details.get("company_name", "Company")
        position = job_details.get("job_title", "Position")

        remove_directory("output")
        os.makedirs("output", exist_ok=True)
        shutil.copy("templates/resume.cls", "output")
        shutil.copy("templates/resume.tex.jinja", "output")

        initial_score = calculate_ats_score(json.dumps(user_data), json.dumps(job_details))

        # Build Resume
        if get_resume_button:
            with st.status("Building resume..."):
                resume_path, resume_details = resume_llm.resume_builder(job_details, user_data, is_st=True)

            resume_col_1, resume_col_2, resume_col_3 = st.columns([0.35, 0.3, 0.25])
            with resume_col_1:
                st.subheader("Generated Resume")
            with resume_col_2:
                pdf_data = read_file(resume_path, "rb")
                st.download_button(label="Download Resume ⬇",
                                   data=pdf_data,
                                   file_name=os.path.basename(resume_path),
                                   mime="application/pdf",
                                   use_container_width=True)
            with resume_col_3:
                # Overleaf button
                create_overleaf_button(resume_path)

            display_pdf(resume_path, type="image")
            st.toast("Resume generated successfully!", icon="✅")

            new_score = calculate_ats_score(json.dumps(resume_details), json.dumps(job_details))
            st.subheader("Resume Metrics")
            # Using overlap_coefficient and cosine_similarity
            for metric in ['overlap_coefficient', 'cosine_similarity']:
                user_personalization = globals()[metric](json.dumps(resume_details), json.dumps(user_data))
                job_alignment = globals()[metric](json.dumps(resume_details), json.dumps(job_details))
                job_match = globals()[metric](json.dumps(user_data), json.dumps(job_details))

                if metric == "overlap_coefficient":
                    title = "Token Space"
                    help_text = "Token space compares texts by exact tokens..."
                elif metric == "cosine_similarity":
                    title = "Latent Space"
                    help_text = "Latent space looks at meaning..."

                st.caption(f"## **:rainbow[{title}]**", help=help_text)
                col_m_1, col_m_2, col_m_3 = st.columns(3)
                col_m_1.metric(label=":green[User Personalization Score]", value=f"{user_personalization:.3f}", delta="(new resume, old resume)", delta_color="off")
                col_m_2.metric(label=":blue[Job Alignment Score]", value=f"{job_alignment:.3f}", delta="(new resume, job details)", delta_color="off")
                col_m_3.metric(label=":violet[Job Match Score]", value=f"{job_match:.3f}", delta="[old resume, job details]", delta_color="off")
            st.markdown("---")

            st.session_state.generated_resume = {
                'path': resume_path,
                'filename': generate_filename(user_name, company, position, "Resume"),
            }

        # Build Cover Letter
        if get_cover_letter_button:
            with st.status("Building cover letter..."):
                cv_details, cv_path = resume_llm.cover_letter_generator(job_details, user_data, is_st=True)

            cv_col_1, cv_col_2 = st.columns([0.7, 0.3])
            with cv_col_1:
                st.subheader("Generated Cover Letter")
            with cv_col_2:
                cv_data = read_file(cv_path, "rb")
                st.download_button(label="Download CV ⬇",
                                   data=cv_data,
                                   file_name=os.path.basename(cv_path),
                                   mime="application/pdf", 
                                   use_container_width=True)
            st.markdown(cv_details, unsafe_allow_html=True)
            st.markdown("---")
            st.toast("Cover letter generated successfully!", icon="✅")

            st.session_state.generated_cover_letter = {
                'path': cv_path,
                'filename': generate_filename(user_name, company, position, "Cover_Letter"),
                'details': cv_details,
            }

        st.toast("Done", icon="👍🏻")
        st.success("Done", icon="👍🏻")
        st.balloons()

        application_entry = {
            'company': company,
            'position': position,
            'date': job_details.get("extraction_date", ""),
            'resume': st.session_state.generated_resume,
            'cover_letter': st.session_state.generated_cover_letter,
            'status': "Resume & Cover Letter Generated"
        }
        st.session_state.applications.append(application_entry)

        refresh = st.button("Refresh")
        if refresh:
            st.experimental_rerun()

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

st.link_button("Report Feedback, Issues, or Contribute!", "https://github.com/Ztrimus/job-llm/issues", use_container_width=True)
