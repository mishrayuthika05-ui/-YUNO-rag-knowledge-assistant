# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) Knowledge Assistant built using FastAPI, LangChain, Ollama, Phi3, FAISS, and Sentence Transformers.

## Features

* Document ingestion and preprocessing
* Semantic search using vector embeddings
* FAISS-powered vector database
* Retrieval-Augmented Generation (RAG)
* Source-aware responses
* FastAPI REST API backend
* Local LLM inference with Ollama

## Tech Stack

* Python
* FastAPI
* LangChain
* Ollama
* Phi3 Mini
* FAISS
* Sentence Transformers

## Project Structure

AI-assistent/

├── modules/

│ ├── agent.py

│ ├── embeddings.py

│ ├── ingestion.py

│ ├── rag_chain.py

│ └── retriever.py

├── tests/

├── main/

│ └── main.py

├── store.faiss

├── README.md

├── requirements.txt

└── .gitignore

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
uvicorn main.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

## Architecture

Document
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
Phi3 LLM
↓
Answer + Sources

## Future Improvements

* Multi-document support
* PDF ingestion
* Streamlit chat interface
* Conversation memory
* Hybrid search
