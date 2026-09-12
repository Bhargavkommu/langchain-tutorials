"""
Chapter 08 — Vector Stores (FAISS deep dive)
=============================================
Goal: Understand how vector stores work — create, save to disk, load, and search.

What you learn:
- Document: LangChain's standard wrapper for a piece of text + optional metadata
- FAISS.from_documents(): converts documents to vectors and stores them
- save_local(): persists the vector store to disk (avoids re-embedding on restart)
- load_local(): loads a saved vector store from disk
- similarity_search(query, k=N): finds the N most semantically similar documents

Why save to disk?
  Embedding is slow and uses compute resources.
  In production: embed once -> save -> load on every app restart (much faster).

Note: similarity_search uses vector math only — no LLM involved.
  The result is based on MEANING similarity, not keyword matching.

Run: python chapter-08-vector-stores/vector_store.py
"""

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create documents manually
# In real projects these come from files, databases, APIs, or web scraping
# Each Document has page_content (text) and optional metadata (source, date, etc.)
docs = [
    Document(page_content="Python is a popular programming language known for its simplicity."),
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="FAISS is a library for efficient similarity search of vectors."),
    Document(page_content="Ollama lets you run large language models locally on your machine."),
]

# Convert all documents to vectors and store in FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(docs, embeddings)

# Save to disk — creates a folder with two files:
#   index.faiss  -> the actual vectors
#   index.pkl    -> document metadata
vector_store.save_local("chapter-08-vector-stores/faiss_index")
print("Vector store saved!")

# Load from disk — no re-embedding needed
# allow_dangerous_deserialization=True is required to load .pkl files
loaded_store = FAISS.load_local(
    "chapter-08-vector-stores/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Similarity search: find the 2 most semantically similar documents to the query
# No LLM involved — this is pure vector math (cosine similarity)
results = loaded_store.similarity_search("What is LangChain?", k=2)

for doc in results:
    print("Retrieved:", doc.page_content)
