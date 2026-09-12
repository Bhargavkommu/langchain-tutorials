# LangChain & LangGraph Tutorials

A hands-on, step-by-step tutorial series for learning LangChain and LangGraph from scratch.
Built using **Ollama (llama3.2)** locally and **IBM watsonx** for cloud LLM access.

## Who is this for?
- Beginners to Python and LLM frameworks
- Anyone who wants to build real AI applications with LangChain

## What you will learn

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01 | Setup & First LLM Call | IBM watsonx, Credentials, ModelInference |
| 02 | Prompt Templates | PromptTemplate, ChatPromptTemplate |
| 03 | Chains (LCEL) | pipe operator, StrOutputParser |
| 04 | Output Parsers | JsonOutputParser, PydanticOutputParser |
| 05 | Memory & Chat History | InMemoryChatMessageHistory, RunnableWithMessageHistory |
| 06 | Tools & Agents | @tool decorator, create_react_agent, ReAct loop |
| 07 | RAG Basics | TextLoader, FAISS, OllamaEmbeddings, retrieval chain |
| 08 | Vector Stores | FAISS save/load, similarity_search |
| 09 | Full RAG App | End-to-end RAG with interactive loop |
| 10 | LangGraph Intro | StateGraph, nodes, edges, compile |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Bhargavkommu/langchain-tutorials.git
cd langchain-tutorials

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install langchain langchain-ibm langchain-ollama langchain-community \
            langchain-text-splitters langchain-core faiss-cpu \
            python-dotenv ibm-watsonx-ai langgraph pydantic

# 4. Copy and fill in your credentials (only needed for chapters 01-02)
cp .env.example .env
```

## Running a chapter

```bash
# Always run from the repo root folder
python chapter-01-setup/hello_world.py
python chapter-03-chains/chains.py
python chapter-09-full-rag/rag_app.py
```

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed locally with `llama3.2` and `nomic-embed-text` pulled
- IBM watsonx credentials (only for chapters 01-02)
