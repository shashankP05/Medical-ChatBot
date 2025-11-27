# Medical ChatBot

An AI-powered medical information chatbot built with Flask, LangChain, and Google Gemini API. The chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to medical queries by retrieving relevant information from a vector database.

<img width="1229" height="782" alt="Screenshot 2025-11-21 160133" src="https://github.com/user-attachments/assets/54dfb1fa-8f00-49fc-8105-4291056cc80b" />

## Features

- Natural language medical question answering
- Context-aware responses using RAG architecture
- Vector-based semantic search with Pinecone
- Google Gemini API for text generation
- Real-time chat interface
- Efficient embedding model using sentence-transformers

## Tech Stack

### Backend
- **Flask** - Web framework
- **LangChain** - LLM orchestration and RAG pipeline
- **Google Gemini API** - Large language model for response generation
- **Pinecone** - Vector database for semantic search
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **Gunicorn** - Production WSGI server

### Frontend
- HTML/CSS
- Vanilla JavaScript
- Responsive chat interface

## Architecture

The application follows a RAG (Retrieval-Augmented Generation) architecture:

1. User submits a medical query
2. Query is converted to embeddings using Sentence Transformers
3. Pinecone retrieves relevant medical context from vector database
4. Context and query are combined into a prompt
5. Google Gemini generates an accurate, contextual response
6. Response is returned to the user interface

## Prerequisites

- Python 3.10 or higher
- Pinecone account and API key
- Google Cloud account with Gemini API access
- Conda or pip for package management
