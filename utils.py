def clean_text(text: str) -> str:
    """Basic text cleaning for resumes and job descriptions."""
    text = text.replace("\n", " ").strip()
    return text

def combine_inputs(resume: str, job: str) -> str:
    """Combine resume and job description for classification."""
    return clean_text(resume) + " " + clean_text(job)
 
