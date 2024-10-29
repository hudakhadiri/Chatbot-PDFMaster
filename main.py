import streamlit as st
from transformers import pipeline
from langchain_core.messages import AIMessage, HumanMessage


context = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving".
"""

# Load pre-trained model and tokenizer from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Streamlit app
st.title("Simple Chatbot  :wolf:")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
             AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

user_input = st.chat_input("Type your message here...")
if user_input is not None and user_input != "":
    result = qa_pipeline(question=user_input, context=context)
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


