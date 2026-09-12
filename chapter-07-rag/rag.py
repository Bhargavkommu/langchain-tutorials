"""
Chapter 07 — RAG Basics (Retrieval-Augmented Generation)
=========================================================
Goal: Feed your own document to the LLM so it can answer questions about it.

What you learn:
- TextLoader: reads a .txt file into LangChain Document objects
- RecursiveCharacterTextSplitter: splits long text into smaller chunks
- OllamaEmbeddings: converts text chunks into vectors (numbers representing meaning)
- FAISS: stores vectors and enables fast similarity search
- Retrieval chain: finds relevant chunks and sends them to the LLM with the question

RAG Pipeline:
  data.txt -> load -> split into chunks -> embed -> store in FAISS
  Question -> embed -> search FAISS -> get relevant chunks -> LLM -> answer

IMPORTANT: Use 'nomic-embed-text' for embeddings.
  llama3.2 does NOT support embeddings.
  Run: ollama pull nomic-embed-text

Run: python chapter-07-rag/rag.py
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load the document
loader = TextLoader("chapter-07-rag/data.txt")
docs = loader.load()

# Step 2: Split into chunks
# chunk_size=200: each chunk is max 200 characters
# chunk_overlap=20: chunks share 20 chars with neighbors to preserve context
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Step 3: Embed chunks and store in FAISS vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 4: Create a retriever that searches for relevant chunks
retriever = vector_store.as_retriever()

llm = ChatOllama(model="llama3.2")

# Step 5: Define the RAG prompt
# The LLM receives both the retrieved context and the user's question
prompt = ChatPromptTemplate.from_template(
    "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"
)

# Step 6: Build the LCEL retrieval chain
# retriever fetches relevant chunks, RunnablePassthrough passes the question through
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke("Where is IBM headquartered?")
print(response)
