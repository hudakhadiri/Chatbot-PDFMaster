import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into paragraphs
def split_text_into_paragraphs(text):
    paragraphs = text.split('\n\n')  # Assuming paragraphs are separated by double newlines
    return [para.strip() for para in paragraphs if para.strip()]

# Load pre-trained model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")

# Create embeddings for text chunks
def create_embeddings(text_chunks):
    inputs = tokenizer(text_chunks, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Streamlit app
st.title("PDF Chatbot")

# Upload PDF
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

# Extract text and create embeddings
if pdf_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(pdf_file)
        text_chunks = split_text_into_paragraphs(text)
        embeddings = create_embeddings(text_chunks)
    
    # User input
    user_input = st.text_input("Ask me something about the PDF")
    
    if user_input:
        user_input_embedding = create_embeddings([user_input])[0]
        similarities = cosine_similarity([user_input_embedding], embeddings)[0]
        relevant_chunk_indices = similarities.argsort()[-3:][::-1]
        relevant_chunks = [text_chunks[i] for i in relevant_chunk_indices]
        
        # Generate response by combining relevant chunks
        response = "\n\n".join(relevant_chunks[:2])  # Limiting to top 2 relevant chunks for clarity
        st.write(response)

    # Clear embeddings after use to prevent repetitive answers
    embeddings = []
