import import
import streamlit as st
from transformers import pipeline
from langchain_core.messages import AIMessage, HumanMessage
import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load pre-trained model and tokenizer from Hugging Face
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def get_most_relevant_chunk(question, context, max_length=512):
    # Split context into chunks
    chunks = [context[i:i + max_length] for i in range(0, len(context), max_length)]
    best_chunk = None
    best_score = -1
    # Find the best chunk based on the model's score
    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result['score'] > best_score:
            best_score = result['score']
            best_chunk = chunk
    return best_chunk

# Streamlit app
st.title("Simple Chatbot :wolf:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.session_state.pdf_text = pdf_text
    st.write("PDF loaded successfully!")

user_input = st.chat_input("Type your message here...")

if user_input is not None and user_input != "":
    if "pdf_text" in st.session_state:
        context = st.session_state.pdf_text
        relevant_chunk = get_most_relevant_chunk(user_input, context)
        result = qa_pipeline(question=user_input, context=relevant_chunk)
        response = result['answer']
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
