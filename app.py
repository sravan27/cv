import os
import json
import shutil
import nltk
import streamlit as st
from datetime import datetime
from zlm import AutoApplyModel
from zlm.utils.utils import read_file
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity
from openai import AzureOpenAI

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-03-15-preview",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"]
)
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

st.set_page_config(
    page_title="AI Career Booster",
    page_icon="🚀",
    layout="wide",
    menu_items={
        'Get help': 'https://github.com/hireopt/zlm',
        'About': 'https://github.com/hireopt/zlm',
        'Report a bug': "https://github.com/hireopt/zlm/issues",
    }
)

def calculate_ats_score(text1, text2):
    score = (
        jaccard_similarity(text1, text2) * 0.4
        + overlap_coefficient(text1, text2) * 0.3
        + cosine_similarity(text1, text2) * 0.3
    )
    return round(score * 100, 2)

def extract_job_details(job_description, model):
    job_details = model.job_details_extraction(job_site_content=job_description, is_st=True)[0]
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
    return f"{safe_name}_{safe_company}_{safe_position}_{doc_type}_{timestamp}.json"

def interview_system_prompt(user_data, job_details):
    return (
        f"You are a professional interviewer and career coach with deep knowledge of the company's needs and the candidate's background.\n"
        f"Context:\n"
        f"Job Description:\n{json.dumps(job_details, indent=2)}\n"
        f"Candidate Resume Data:\n{json.dumps(user_data, indent=2)}\n"
        f"Your Role:\n"
        f"1. Ask one challenging, context-specific interview question at a time.\n"
        f"2. After the candidate responds, provide constructive, personalized feedback focusing on areas to improve, and highlight strengths.\n"
        f"3. Coach the candidate by referencing both the job needs and their experience.\n"
        f"4. Then ask the next question, going deeper each time.\n"
        f"Be professional, encouraging, yet challenging."
    )

def generate_interview_turn(user_message, job_description, user_data, job_details):
    messages = [{"role": "system", "content": interview_system_prompt(user_data, job_details)}]
    if job_description:
        messages.append({"role": "system", "content": f"Additional Job Description:\n{job_description}"})
    messages.extend(st.session_state.interview_history)
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

def remove_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)

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
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'job_details' not in st.session_state:
    st.session_state.job_details = None

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the mode:", ["Resume & Cover Letter Generator", "Interview Preparation", "Upcoming Features"])

if app_mode == "Resume & Cover Letter Generator":
    st.header("Get a Job-Aligned, Personalized Resume and ATS Scores")
    st.write("Upload your current resume as a PDF. We'll analyze it and the job description to give you ATS scores.")
    job_description = st.text_area("Paste job description:", max_chars=5500, height=200, placeholder="Paste JD here...")
    resume_file = st.file_uploader("Upload your current resume (PDF)", type=["pdf"])
    generate_resume_button = st.button("Generate Optimized Resume Data (JSON) and ATS Score Improvement")

    if generate_resume_button and job_description and resume_file:
        api_key = st.secrets["OPENAI_API_KEY"]
        resume_llm = AutoApplyModel(api_key=api_key, provider="GPT", model="gpt-4o", downloads_dir="output")
        os.makedirs("uploads", exist_ok=True)
        resume_path = os.path.join("uploads", resume_file.name)
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())

        with st.spinner("Extracting user data from PDF resume..."):
            user_data = resume_llm.user_data_extraction(resume_path, is_st=True)
            st.session_state.user_data = user_data

        shutil.rmtree("uploads")

        if user_data is None:
            st.error("Could not extract user data from the uploaded resume. Please try again with a valid PDF resume.")
            st.stop()

        with st.spinner("Extracting job details..."):
            job_details, company, position = extract_job_details(job_description, resume_llm)
            st.session_state.job_details = job_details

        initial_score = calculate_ats_score(json.dumps(user_data), json.dumps(job_details))

        with st.spinner("Generating optimized resume data (no PDF generation, just JSON) ..."):
            _, optimized_resume_details = resume_llm.resume_builder(job_details, user_data, is_st=True)

        new_score = calculate_ats_score(json.dumps(optimized_resume_details), json.dumps(job_details))

        st.subheader("ATS Score Comparison")
        c1, c2 = st.columns(2)
        c1.metric("Original ATS Score", f"{initial_score}%")
        c2.metric("Optimized ATS Score", f"{new_score}%", delta=f"{new_score - initial_score}%")

        st.subheader("Optimized Resume Data (JSON)")
        st.json(optimized_resume_details)

        application_entry = {
            'company': company,
            'position': position,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'resume': {'json_data': optimized_resume_details},
            'cover_letter': None,
            'status': "Resume Generated"
        }
        st.session_state.applications.append(application_entry)

        st.success("ATS Score improved successfully!")

    if st.session_state.applications:
        st.subheader("Job Application Tracker")
        stages = ["Resume Generated", "Applied", "Interviewed", "Offer Received", "Hired", "Rejected"]
        for idx, app in enumerate(st.session_state.applications):
            with st.expander(f"Application {idx + 1}: {app['position']} at {app['company']} ({app['date']})"):
                current_status = app['status']
                app['status'] = st.selectbox("Application Status:", stages, index=stages.index(current_status), key=f"app_status_{idx}")
                st.write("Optimized Resume Data:")
                if app['resume'] and app['resume'].get('json_data'):
                    st.json(app['resume']['json_data'])
                else:
                    st.info("No resume data available.")

elif app_mode == "Interview Preparation":
    st.header("AI Interview Preparation")
    st.write("This uses the previously extracted resume data and job details.")
    if st.session_state.job_details and st.session_state.user_data:
        st.write("We have the job details and your resume data. Let's get started.")
    else:
        st.warning("No job description and resume data found. Please go to Resume & Cover Letter Generator first.")
        st.stop()

    interview_type = st.selectbox("Select Interview Type", ["General", "HR Round", "Technical Round", "Behavioral Round"])
    generate_interview = st.button("Generate Initial Interview Questions")

    if generate_interview:
        with st.spinner("Generating initial interview questions..."):
            system_prompt = f"You are a helpful assistant generating {interview_type} interview questions."
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate 5 {interview_type} interview questions for the given job description."}
                ],
                temperature=0.7,
                max_tokens=500
            )
            interview_questions = response.choices[0].message.content
            st.session_state.interview_questions = interview_questions
            st.text_area("Suggested Interview Questions:", value=interview_questions, height=200)

    st.markdown("---")
    st.subheader("Interactive Interview & Coaching Session")
    start_interview = st.button("Start Interview", disabled=st.session_state.interview_active or not (st.session_state.job_details and st.session_state.user_data))
    if start_interview:
        st.session_state.interview_history = []
        st.session_state.interview_active = True
        initial_question = "Let's begin. Based on the job description and your experience, could you describe how your key accomplishments align with the role's requirements?"
        st.session_state.interview_history.append({"role": "assistant", "content": initial_question})
        st.success("Interview session started!")
        st.write(f"**Interviewer:** {initial_question}")

    if st.session_state.interview_active:
        user_answer = st.text_area("Your answer:", height=100, placeholder="Type your response here...")
        answer_submitted = st.button("Send Answer")
        if answer_submitted and user_answer.strip():
            st.session_state.interview_history.append({"role": "user", "content": user_answer})
            response = generate_interview_turn(user_answer, st.session_state.job_description, st.session_state.user_data, st.session_state.job_details)
            st.session_state.interview_history.append({"role": "assistant", "content": response})
            st.write(f"**Interviewer/Coach:** {response}")
        stop_interview = st.button("Stop Interview")
        if stop_interview:
            st.session_state.interview_active = False
            st.success("Interview session ended.")

elif app_mode == "Upcoming Features":
    st.header("Upcoming Features & PDF Generation Explanation")
    st.write("Future enhancements:")
    st.markdown("""
    - PDF Generation (currently omitted due to environment restrictions)
    - Multimodal Integration (images, audio)
    - Advanced Provider Management
    - Batch Applications
    """)
    st.write("We currently do not generate PDF resumes due to environment constraints. Once dependencies are available, we'll enable polished PDF outputs.")
    st.subheader("Preview of Multimodal Integration")
    multimodal_file = st.file_uploader("Upload an asset:", type=["png", "jpg", "jpeg"])
    if multimodal_file:
        st.warning("Multimodal integration is under development. The uploaded file won't affect current ATS scoring or interview prep.")

st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)
st.caption("Built on top of zlm by hireopt and team.")
