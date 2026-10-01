# Gemma Model Q&A Chatbot

A Streamlit-based Q&A chatbot that reads PDF documents, converts them into searchable vector embeddings, and answers user questions based on the document content using the Gemma model via Groq.

## Overview

This project is a lightweight Retrieval-Augmented Generation (RAG) application built with:
- Python
- Streamlit
- LangChain
- Groq
- FAISS
- Google Generative AI embeddings
- PyPDF

It allows users to upload or place PDF files in a folder, process them, and ask natural-language questions about their content.

## Features

- PDF document loading from a local folder
- Text chunking and splitting for better retrieval
- Vector embedding generation with Google Generative AI
- FAISS-based similarity search
- Question answering using Groq-hosted Gemma model
- Simple Streamlit interface for interaction

## Tech Stack

- Python 3.8+
- Streamlit
- LangChain
- LangChain Groq integration
- LangChain Community FAISS vector store
- PyPDFDirectoryLoader
- Google Generative AI Embeddings

## Repository Structure

```bash
chatbot_gemma/
├── main.py
├── requirements.txt
├── .env
├── pdf_docs/
│   └── your_pdf_files_here.pdf
└── README.md
