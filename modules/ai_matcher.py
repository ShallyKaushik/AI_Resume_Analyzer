# modules/ai_matcher.py
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from transformers import pipeline

# We use st.cache_resource so the heavy BERT model only loads once per session
@st.cache_resource
def load_bert_model():
    """Loads the pre-trained BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def get_bert_embedding(text, tokenizer, model):
    """Converts text into a BERT embedding vector."""
    if not text:
        return None
        
    # Tokenize and encode the text (handling length limits)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Extract the [CLS] token's embedding to represent the entire text block
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

def calculate_ats_score(resume_text, job_desc_text):
    """Calculates the semantic similarity score between resume and job description."""
    tokenizer, model = load_bert_model()
    
    resume_emb = get_bert_embedding(resume_text, tokenizer, model)
    job_desc_emb = get_bert_embedding(job_desc_text, tokenizer, model)
    
    if resume_emb is None or job_desc_emb is None:
        return 0.0
        
    # Calculate Cosine Similarity using Scikit-Learn
    similarity = cosine_similarity(resume_emb, job_desc_emb)
    
    # Convert to a percentage score
    score_percentage = round(similarity[0][0] * 100, 2)
    
    # Ensure the score stays within bounds
    return max(0.0, min(score_percentage, 100.0))

@st.cache_resource
def load_classifier():
    """Loads a pre-trained transformer model for zero-shot classification."""
    # Using a standard robust model for zero-shot classification
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_resume(resume_text):
    """Classifies the resume into standard tech job categories."""
    if not resume_text:
        return "Unknown", 0.0
        
    classifier = load_classifier()
    
    # Define the categories you want the AI to choose from
    candidate_labels = [
        "Software Engineering", 
        "Data Science", 
        "Web Development", 
        "Machine Learning", 
        "Cloud Computing", 
        "Product Management"
    ]
    
    # The model will return the labels ranked by probability
    result = classifier(resume_text, candidate_labels)
    
    # Return the top predicted category and its confidence score
    top_category = result['labels'][0]
    confidence = round(result['scores'][0] * 100, 2)
    
    return top_category, confidence