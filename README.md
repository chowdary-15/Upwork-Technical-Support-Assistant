# Upwork Technical Support Assistant

A Retrieval-Augmented Generation (RAG) chatbot built to answer developer questions using the Upwork Enterprise API documentation. The assistant retrieves relevant documentation sections from a local vector database and uses an LLM to generate grounded responses while reducing hallucinations.

## Live Demo

**Application:** https://upwork-technical-support-assistant.streamlit.app/

**GitHub Repository:** https://github.com/chowdary-15/Upwork-Technical-Support-Assistant

---

## Project Overview

This project was developed as a technical support assistant for developers working with the Upwork Enterprise API.

Instead of relying solely on a language model's internal knowledge, the application retrieves relevant content from official documentation and uses that context to answer user questions accurately.

The system supports questions related to:

* OAuth 2.0 authentication
* Access and refresh tokens
* GraphQL APIs
* API endpoints
* Authorization scopes
* Upwork Enterprise platform documentation

---

## Features

### Document Processing

* Loads documentation from PDF files using PyPDF.
* Extracts content page by page.
* Preserves technical information through intelligent chunking.

### Retrieval-Augmented Generation (RAG)

* Converts document chunks into vector embeddings.
* Stores embeddings locally using FAISS.
* Retrieves the most relevant documentation sections for every query.

### Hallucination Reduction

* Uses retrieved documentation as the primary source of truth.
* Includes prompt guardrails to reduce unsupported answers.
* Returns fallback responses when sufficient context is unavailable.

### Streamlit User Interface

* Clean and responsive web interface.
* Sidebar chat history.
* One-click chat reset functionality.

### Local Vector Store

* FAISS-based vector database.
* Automatic loading of existing indexes.
* Rebuilds indexes when required.

---

## System Architecture

1. PDF documentation is loaded using PyPDF.
2. Documentation is split into chunks using RecursiveCharacterTextSplitter.
3. Chunks are converted into embeddings using Sentence Transformers.
4. Embeddings are stored in a FAISS vector database.
5. User queries trigger semantic similarity search.
6. Relevant context is injected into the LLM prompt.
7. The LLM generates a response using only retrieved documentation.

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### RAG Components

* LangChain
* FAISS
* Sentence Transformers

### LLM Integration

* DeepInfra API
* OpenAI SDK Interface

### Document Processing

* PyPDF

---

## Project Structure

```text
Upwork-Technical-Support-Assistant/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env.example
├── API Documentation Partial.pdf
└── faiss_upwork_index/
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/chowdary-15/Upwork-Technical-Support-Assistant.git

cd Upwork-Technical-Support-Assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

```env
DEEPINFRA_API_KEY=your_api_key

DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically:

* Load an existing FAISS index if available.
* Process the PDF if no index exists.
* Build embeddings.
* Launch the chatbot interface.

---

## Challenges Solved

### Context Preservation

Technical documentation often contains related information spread across multiple sections. Chunk overlap was used to maintain context across document boundaries.

### Dependency Management

Resolved compatibility issues involving:

* TensorFlow
* Keras
* Protobuf
* Sentence Transformers

to ensure reliable local and cloud deployment.

### Hallucination Control

The assistant is instructed to answer only from retrieved documentation context rather than generating unsupported information.



