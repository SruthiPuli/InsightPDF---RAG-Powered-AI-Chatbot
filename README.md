# InsightPDF - RAG-Powered AI Chatbot

A **PDF-based Retrieval-Augmented Generation (RAG) chatbot** allows users to interact with PDF documents intelligently. Instead of manually searching through pages, you can simply ask questions, and the chatbot extracts relevant information from your PDFs, providing accurate and context-aware answers instantly.

This chatbot built with **Streamlit, LangChain, HuggingFace embeddings, FAISS, and Groq LLM.**

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Screenshots](#screenshots)
7. [Technologies Used](#technologies-used)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

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
