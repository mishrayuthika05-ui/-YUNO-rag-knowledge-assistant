# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) Knowledge Assistant built using:

- FastAPI
- LangChain
- Ollama
- Phi3
- FAISS
- Sentence Transformers

## Features

- Document ingestion
- Semantic search
- FAISS vector database
- RAG-based question answering
- FastAPI backend

## Run

```bash
uvicorn main:app --reload
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
FAISS
↓
Retriever
↓
LLM
↓
Answer + Sources
