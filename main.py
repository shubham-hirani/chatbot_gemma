import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.title("Gemma Model Q&A Chatbot")

llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],
                model_name='gemma2-9b-it')
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate answers only based on the question.
<context>
{context}
<context>
Questions: {input}
""")


def vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        st.session_state.loader = PyPDFDirectoryLoader('./pdf_docs')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


prompt1 = st.text_input("What do you want to know?")

if st.button("Creating vectors & Processing PDFs"):
    vector_embeddings()
    st.write("Vector is created!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response['answer'])