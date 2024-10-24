import streamlit as st
import pandas as pnd



def main():

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("ask question")

    with st.sidebar:
        st.subheader("your docs")
        st.file_uploader("upload file")
        st.button("process")


if __name__ == '__main__':
    main()
