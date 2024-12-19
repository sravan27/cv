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

def interview_system_prompt(user_data, job_details):
    return (
        "You are a professional interviewer and career coach with deep knowledge of the company's needs and the candidate's background.\n"
        "You have the following context:\n"
        f"Job Description:\n{json.dumps(job_details, indent=2)}\n\n"
        f"Candidate Resume Data:\n{json.dumps(user_data, indent=2)}\n\n"
        "Your Role:\n"
        "1. Ask one challenging, context-specific interview question at a time, leveraging the job requirements and candidate's experience.\n"
        "2. After the candidate responds, provide constructive, personalized feedback focusing on their answer, guiding improvements and highlighting strengths.\n"
        "3. Then ask the next question, each time going deeper, ensuring the candidate is truly prepared.\n"
        "Be professional, encouraging, yet challenging. Make the questions relevant to the job and candidate. Incorporate details from the resume and job description to keep it highly contextual."
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
app_mode = st.sidebar.selectbox("Choose the app mode", ["Resume & Cover Letter Generator", "Interview Preparation", "Upcoming Features"])

if app_mode == "Resume & Cover Letter Generator":
    st.header("📑 Job-Aligned Resume & Cover Letter Generation with ATS Analysis")
    st.write("Upload your resume (PDF) and job description. We'll analyze and improve your resume (JSON suggestions), provide a cover letter draft, and show ATS score improvements.")

    job_description = st.text_area("Paste Job Description:", max_chars=5500, height=200, placeholder="Paste the job description here...")
    resume_file = st.file_uploader("Upload your current Resume (PDF):", type=["pdf"])

    col_actions = st.columns(3)
    with col_actions[1]:
        generate_button = st.button("Generate Optimized Resume & Cover Letter")

    if generate_button and job_description and resume_file:
        api_key = st.secrets["OPENAI_API_KEY"]
        resume_llm = AutoApplyModel(api_key=api_key, provider="GPT", model="gpt-4o", downloads_dir="output")

        os.makedirs("uploads", exist_ok=True)
        resume_path = os.path.join("uploads", resume_file.name)
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())

        with st.spinner("Extracting user data from the PDF resume..."):
            user_data = resume_llm.user_data_extraction(resume_path, is_st=True)
            st.session_state.user_data = user_data

        remove_directory("uploads")

        if user_data is None:
            st.error("Unable to extract data from the uploaded PDF. Please ensure it's a valid resume.")
            st.stop()

        with st.spinner("Extracting job details..."):
            job_details, company, position = extract_job_details(job_description, resume_llm)
            st.session_state.job_details = job_details

        initial_score = calculate_ats_score(json.dumps(user_data), json.dumps(job_details))

        with st.spinner("Generating optimized resume data (JSON suggestions)..."):
            _, resume_details = resume_llm.resume_builder(job_details, user_data, is_st=True)

        new_score = calculate_ats_score(json.dumps(resume_details), json.dumps(job_details))

        with st.spinner("Generating cover letter draft..."):
            cover_letter_details, _ = resume_llm.cover_letter_generator(job_details, user_data, is_st=True)

        st.subheader("🎯 ATS Score Comparison")
        col1, col2 = st.columns(2)
        col1.metric("Original ATS Score", f"{initial_score}%")
        col2.metric("Optimized ATS Score", f"{new_score}%", delta=f"{new_score - initial_score}%")

        st.subheader("📝 Optimized Resume Suggestions (JSON)")
        st.write("Below is the optimized resume data in JSON format. You can copy and adapt it in your own resume template.")
        st.json(resume_details)

        st.subheader("✍️ Cover Letter Draft")
        st.write("Here is a suggested cover letter. You can copy and refine it before sending to the employer.")
        st.text_area("Copy-Paste Your Cover Letter:", value=cover_letter_details, height=300)

        application_entry = {
            'company': company,
            'position': position,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'resume': {'json_data': resume_details},
            'cover_letter': {'text': cover_letter_details},
            'status': "Resume & Cover Letter Generated"
        }
        st.session_state.applications.append(application_entry)

        st.success("Optimized resume and cover letter generated. ATS Score improved!")

    if st.session_state.applications:
        st.subheader("📜 Job Application Tracker")
        stages = ["Resume & Cover Letter Generated", "Applied", "Interviewed", "Offer Received", "Hired", "Rejected"]
        for idx, app in enumerate(st.session_state.applications):
            with st.expander(f"Application {idx+1}: {app['position']} at {app['company']} ({app['date']})"):
                app['status'] = st.selectbox("Application Status:", stages, index=stages.index(app['status']), key=f"app_status_{idx}")
                st.write("**Optimized Resume Data:**")
                if app['resume'] and app['resume'].get('json_data'):
                    st.json(app['resume']['json_data'])
                else:
                    st.info("No resume data available.")
                st.write("**Cover Letter Draft:**")
                if app['cover_letter'] and app['cover_letter'].get('text'):
                    st.text_area("Cover Letter:", value=app['cover_letter']['text'], height=200)
                else:
                    st.info("No cover letter available.")

elif app_mode == "Interview Preparation":
    st.header("🎤 AI Interview Preparation")
    st.write("Leverages your resume and job description to create challenging, context-aware interview questions. After each response, you'll receive feedback and a follow-up question.")

    if st.session_state.job_details and st.session_state.user_data:
        st.write("We have the job description and your resume data loaded.")
    else:
        st.warning("No job description and resume data found. Please return to 'Resume & Cover Letter Generator' first.")
        st.stop()

    interview_type = st.selectbox("Select Interview Type:", ["General", "HR Round", "Technical Round", "Behavioral Round"])
    col_interview = st.columns(3)
    with col_interview[1]:
        generate_interview = st.button("Generate Initial Interview Questions")

    if generate_interview:
        with st.spinner("Generating initial interview questions..."):
            system_prompt = f"You are a helpful assistant generating {interview_type} interview questions."
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate 5 {interview_type} interview questions tailored to the given job description."}
                ],
                temperature=0.7,
                max_tokens=500
            )
            interview_questions = response.choices[0].message.content
            st.session_state.interview_questions = interview_questions
            st.subheader("Suggested Interview Questions")
            st.text_area("These questions are suggested starting points:", value=interview_questions, height=200)

    st.markdown("---")
    st.subheader("🤝 Interactive Interview & Coaching Session")
    st.write("Click 'Start Interview' to begin. You'll receive a question, respond, then get feedback and the next question.")
    start_interview = st.button("Start Interview", disabled=st.session_state.interview_active or not (st.session_state.job_details and st.session_state.user_data))
    if start_interview:
        st.session_state.interview_history = []
        st.session_state.interview_active = True
        initial_question = "Let's begin. Considering the job description and your experience, how do your key accomplishments align with the role's core requirements?"
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
    st.header("🚀 Upcoming Features & PDF Generation Explanation")
    st.markdown("""
    - **PDF Generation:** Once environment dependencies are met, we'll provide polished, downloadable PDF resumes.
    - **Multimodal Integration:** We'll allow uploading images, audio, etc., incorporating their content into ATS scoring and interview prep.
    - **Advanced Provider Management:** More control over LLM providers and models.
    - **Batch Applications:** Manage multiple applications, set reminders, and track each stage efficiently.
    """)

    st.write("PDF generation isn't currently provided due to environment restrictions. Once available, you'll get a fully polished PDF resume.")
    st.subheader("Preview of Multimodal Integration")
    multimodal_file = st.file_uploader("Upload an asset (image) to integrate in future:", type=["png","jpg","jpeg"])
    if multimodal_file:
        st.warning("This feature is under development and not yet integrated.")

st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)
st.caption("Built on top of zlm by hireopt and team. Elevate your career with AI-driven insights.")
