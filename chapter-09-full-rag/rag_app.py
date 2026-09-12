"""
Chapter 09 — Full RAG Application
===================================
Goal: Combine everything from chapters 07 & 08 into a complete, interactive RAG app.

What you learn:
- End-to-end RAG pipeline in one clean file
- search_kwargs={"k": 3}: retrieve top 3 most relevant chunks per question
- Grounded prompt: instructs the LLM to ONLY use the provided context
  and say "I don't know" if the answer is not there — prevents hallucination
- Interactive loop: ask multiple questions in one session

Grounding explained:
  Without grounding: LLM uses training data + context (may make things up)
  With grounding:    LLM only uses what's in your document
  Test it: ask about something NOT in knowledge.txt — it will say "I don't know"

Run: python chapter-09-full-rag/rag_app.py
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Step 1: Load the document ---
loader = TextLoader("chapter-09-full-rag/knowledge.txt")
docs = loader.load()

# --- Step 2: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# --- Step 3: Embed and store in FAISS ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)

# --- Step 4: Retriever — fetch top 3 relevant chunks per question ---
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- Step 5: LLM ---
llm = ChatOllama(model="llama3.2")

# --- Step 6: Grounded prompt ---
# Key instruction: "only use context" + "say I don't know if not in context"
# This is what prevents the LLM from hallucinating
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

# --- Step 7: Build the full LCEL chain ---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Step 8: Interactive loop ---
print("RAG App ready! Ask anything about the document. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        print("Goodbye!")
        break
    answer = chain.invoke(question)
    print(f"Bot: {answer}\n")
