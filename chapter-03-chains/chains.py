"""
Chapter 03 — Chains (LCEL)
===========================
Goal: Connect a prompt, LLM, and output parser into a single pipeline using LCEL.

What you learn:
- LCEL (LangChain Expression Language): chaining with the | pipe operator
- StrOutputParser: extracts clean text from an AIMessage object
- chain.invoke(): runs the full pipeline in one call
- Using Ollama locally instead of a cloud API (faster for learning)

The pipe | operator:
  prompt | llm | parser
  Each step's output becomes the next step's input automatically.

Run: python chapter-03-chains/chains.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Use Ollama running locally — no API key needed
llm = ChatOllama(model="llama3.2")

# Define the prompt template with a {topic} placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains things in simple terms."),
    ("human", "Explain {topic} in simple terms for a complete beginner.")
])

# Build the chain: prompt -> llm -> parser
# StrOutputParser extracts the plain text string from the AIMessage object
chain = prompt | llm | StrOutputParser()

# Invoke the chain — fills in {topic}, sends to LLM, returns a clean string
response = chain.invoke({"topic": "Python lists"})
print(response)
