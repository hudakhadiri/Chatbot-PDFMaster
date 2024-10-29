import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Initialize Hugging Face model for conversation
model_name = "facebook/blenderbot-400M-distill"  # Lightweight conversational model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to get text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Set up conversation chain with Hugging Face
def get_conversation_chain(vectorstore):
    # Use Hugging Face model in place of OpenAI
    llm = HuggingFaceHub(repo_id="facebook/blenderbot-400M-distill")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Generate bot response using Hugging Face model
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return bot_response

# Streamlit app setup
def main():
    st.set_page_config(page_title="Chatbot with Free LLM", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chatbot with Hugging Face")
    pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state['conversation_chain'] = get_conversation_chain(vectorstore)

    # Get user input and display chat messages
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        with st.spinner("Generating response..."):
            if 'conversation_chain' in st.session_state:
                # Use the conversational retrieval chain
                response = st.session_state['conversation_chain'].run(user_input)
            else:
                # Fall back to a simple generation without retrieval
                response = generate_response(user_input)

            st.write(bot_template.format(response=response), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
