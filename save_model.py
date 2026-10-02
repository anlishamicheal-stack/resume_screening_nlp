from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pretrained model (start with BERT for testing)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Save both into your models folder
model.save_pretrained("D:/resume_screening_nlp/models/resume_model")
tokenizer.save_pretrained("D:/resume_screening_nlp/models/resume_model")
