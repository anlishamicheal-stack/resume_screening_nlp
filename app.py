import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Path to your local model folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "resume_model")

# ✅ Load model and tokenizer directly (no local_files_only here)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Build pipeline
nlp_model = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

# Streamlit UI
st.title("📄 Resume Screening NLP Tool")

resume_text = st.text_area("Paste Resume Text:", height=200)
job_desc = st.text_area("Paste Job Description:", height=200)

if st.button("Run Screening"):
    if resume_text.strip() and job_desc.strip():
        # Combine resume and job description for classification
        combined_input = resume_text + " [SEP] " + job_desc

        # ✅ Run pipeline cleanly
        result = nlp_model(combined_input)

        # Display prediction
        st.write("Prediction:", result)
    else:
        st.warning("Please enter both resume and job description.")
