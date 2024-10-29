import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Hugging Face model and tokenizer
model_name = "gpt2"  # You can replace this with any conversational model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# App configuration
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Streaming bot")

def get_response(user_query, chat_history):
    # Prepare the chat history
    chat_input = " ".join([msg.content for msg in chat_history])
    input_text = f"{chat_input} User question: {user_query}"

    # Tokenize and generate a response
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input handling
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Display AI response
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))
