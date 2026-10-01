import streamlit as st
from transformers import pipeline

# Absolute path to your fine‑tuned model folder
MODEL_PATH = "D:/resume_screening_nlp/models/resume_model"

# Load the model pipeline
nlp_model = pipeline("text-classification", model=MODEL_PATH)

# Streamlit UI
st.title("📄 Resume Screening NLP Tool")

st.write("Paste a resume and a job description below to check if they match.")

resume_text = st.text_area("Resume Text", height=200)
job_desc = st.text_area("Job Description", height=200)

if st.button("Check Match"):
    if resume_text.strip() and job_desc.strip():
        # Combine inputs
        combined_input = resume_text + " " + job_desc
        result = nlp_model(combined_input)

        st.subheader("🔎 Results")
        st.write("Prediction:", result[0]['label'])
        st.write("Confidence Score:", round(result[0]['score'], 3))
    else:
        st.warning("Please enter both resume and job description.")
