import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ✅ Load model directly from Hugging Face Hub
# Replace "bert-base-uncased" with your own uploaded model name if you fine-tuned one
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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

        # Run pipeline
        result = nlp_model(combined_input)

        # Display prediction nicely
        if result and len(result) > 0:
            prediction = result[0]
            st.success(f"Prediction: **{prediction['label']}** "
                       f"(confidence: {prediction['score']:.2f})")
        else:
            st.warning("No prediction returned.")
    else:
        st.warning("Please enter both resume and job description.")
