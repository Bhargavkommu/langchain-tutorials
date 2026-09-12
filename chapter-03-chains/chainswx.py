"""
Chapter 03 — Chains with a different Ollama model (bonus)
==========================================================
Goal: Show that you can swap the model in a chain without changing anything else.

What you learn:
- You can use any Ollama model just by changing the model name string
- The chain structure (prompt | llm | parser) stays exactly the same
- This is the power of LCEL — the chain is model-agnostic

Try swapping "llama3.2" for any other model you have pulled locally.

Run: python chapter-03-chains/chainswx.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Swap the model name to use a different locally available Ollama model
# e.g. "mistral", "gemma2", "phi3" — whatever you have pulled
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains things simply."),
    ("human", "Explain {topic} in simple terms for a complete beginner.")
])

# Identical chain structure to chains.py — only the model name changed
chain = prompt | llm | StrOutputParser()
response = chain.invoke({"topic": "Python lists"})
print(response)
