# InsightPDF - RAG-Powered AI Chatbot

## Project Overview
A **PDF-based Retrieval-Augmented Generation (RAG) chatbot** allows users to interact with PDF documents intelligently. Instead of manually searching through pages, you can simply ask questions, and the chatbot extracts relevant information from your PDFs, providing accurate and context-aware answers instantly.

This chatbot built with **Streamlit, LangChain, HuggingFace embeddings, FAISS, and Groq LLM.**

**Live Demo :** [(https://insightpdf---rag-powered-ai-chatbot.streamlit.app/)](https://insightpdf---rag-powered-ai-chatbot.streamlit.app/)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Screenshots](#screenshots)
7. [Tech Stack](#tech-stack)
8. [Folder Structure](#folder-structure)
9. [Contributions](#contributions)
10. [License](#license)
11. [About](#about)

## Key Features

- **Context-Aware PDF Intelligence**: Engineered for deep, targeted querying, allowing users to extract precise insights instantly without manual skimming or document searching.  

- **Intelligent Text Processing**: Utilizes automated extraction and recursive chunking logic to preserve document hierarchy and maximize retrieval accuracy.  

- **High-Speed Retrieval with FAISS**: Implements local vector storage for optimized similarity searches, ensuring near-instant access to relevant document segments.  

- **Ultra-Fast Inference via Groq**: Delivers real-time, grounded responses by leveraging Groq’s Tensor Streaming Processor (TSP) architecture for industry-leading, deterministic low latency.  

- **Semantic Precision with HuggingFace**: Employs state-of-the-art embedding models to generate high-fidelity semantic representations, ensuring superior search relevance.

## Installation

### Optional: Create a Python Virtual Environment

To avoid package dependency issues, it is recommended to create a virtual environment before installing the required libraries. You can skip this step if you prefer installing packages globally.  

```bash
# Create a virtual environment named 'my_venv'
python -m venv my_venv
```
```bash
# Activate the virtual environment
# On Windows
my_venv\Scripts\activate
```
```bash
# On macOS/Linux
source my_venv/bin/activate
```
### Install Project Dependencies

Install all required dependencies using the following command:
```bash
# python packages
pip install -r requirements.txt
```

## Usage

- Once the setup is complete, start the Streamlit app by running:

```bash
# Run the chatbot
streamlit run app.py
# Open the URL in your browser (usually http://localhost:8501)
```

- Upload any **PDF file** using the file uploader in the Streamlit interface.

- Once the document is processed and indexed, start asking questions through the chat input.

- The **chatbot** retrieves relevant context from the PDF and **answers your queries in real time.**

## Configuration 

### Configure Environment Variables

 Ensure your `.env` file is set up with your Groq API key:
 ```env
# To access Groq LLM
 GROQ_API_KEY="your_actual_api_key_here"
```
Then, in your Python script, load it like this:
```bash
# Python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

### Streamlit Deployment (Important)

If you deploy this application on **Streamlit Cloud**, do **not** use the `.env` file.

1. Go to your app dashboard on **Streamlit Cloud**.
2. Open **Settings → Secrets**.
3. Add your API key in the following format:
 ```toml
 # Secrets
 GROQ_API_KEY = "your_actual_api_key_here"
 ```
## Screenshots

## Tech Stack

- **Python** – Core programming language used for application logic.  
- **Streamlit** – Interactive web framework for building the user interface.  
- **LangChain** – Provides end-to-end components for building **Retrieval-Augmented Generation (RAG)** pipelines, including document loading, text splitting, embedding integration, vector store management, and seamless LLM orchestration.  
- **Groq LLM** – Used for **real-time response generation**, leveraging Groq’s **Tensor Streaming Processor (TSP)** architecture to deliver ultra-fast, deterministic, low-latency inference for context-aware answers.  
- **HuggingFace Embeddings** – Responsible for converting document text and user queries into **semantic vector representations**, enabling accurate similarity-based retrieval.  
- **FAISS** – High-performance vector database for efficient and fast similarity search over embedded document chunks.  
- **PyPdfReader** – Extracts and processes text from PDF documents.  
- **Sentence-Transformers** – Provides pre-trained embedding models; this project uses **`sentence-transformers/all-MiniLM-L6-v2`** for lightweight, high-quality embeddings that balance speed and semantic accuracy.  
- **Python-dotenv** – Manages environment variables securely during local development.

## Folder Structure

```bash
pdf-rag-chatbot/
├─ assets/                # Images
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ sample_pdf             # Sample Pdf to upload
├─ LICENSE                # MIT License
└─ README.md              # README File              
               

```
## Contributions

Contributions are welcome! If you’d like to improve this project, feel free to fork the repository, create a new branch, and submit a pull request. Bug reports, feature requests, and documentation improvements are all appreciated.

## License

This project is licensed under the [**MIT License**](LICENSE).
If you fork or use this project, please give credit by mentioning or pinging me: **Sruthi Pulipati** ([GitHub: SruthiPuli](https://github.com/SruthiPuli)).

## About

This project is solely developed by **Sruthi Pulipati** ([GitHub: SruthiPuli](https://github.com/SruthiPuli)).
