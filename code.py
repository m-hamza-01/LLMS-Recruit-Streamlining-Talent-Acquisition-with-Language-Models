
# This code streamlines the interview process! Upload resumes (PDFs) and ask a conversational AI any questions
# you have about the candidate's experience. This AI, powered by a large language model, analyzes the resume and 
# provides answers, allowing you to skip basic interview questions. Once satisfied, move directly to real-life
# problem-solving questions in a one-on-one interview!

# inorder to run this run the following command in terminal

# python -m streamlit run code.py

import streamlit as st  #web interface
from PyPDF2 import PdfReader # extract pdf files for resume
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm # uses Google Palm LLM to generate text
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS  #stores data in local computer 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

os.environ['GOOGLE_API_KEY'] =  'PASTE YOUR GOOGLE API HERE'


def get_pdf_text(pdf_docs): #extract pdf documents and extract text from them
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text): #split the data into text_chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):# vector store holds text chunks and their embeddings
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):# keeps track of previous questions and answers to improve response
    llm=GooglePalm()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question): # check id the pdf has been uploade or not
    if st.session_state.conversation is None:
        st.error("Please upload the PDF file first.")
        return
     
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)
def main():
    st.set_page_config("UPLOAD YOUR CSV")
    st.header("ASK Questions from your interviewee (ChatBot)💬")

     # Display generic questions
    st.subheader("Generic Questions")
    st.write("- What was your last working expereince?")
    st.write("- Are you part of any extra curricular activities?")
    st.write("- What skills do you have?")


    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("A revolutionizing way to speed up your recruitment process")
        
        pdf_docs = st.file_uploader("Upload the CV pdf file and click on submit ", accept_multiple_files=True,type=["pdf"])
       
        if st.button("Submit"):
            if pdf_docs is None or len(pdf_docs) == 0:
                st.error("Please upload a PDF file.")
            else:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done")



if __name__ == "__main__":
    main()


  