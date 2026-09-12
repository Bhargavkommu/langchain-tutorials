"""
Chapter 01 — Setup & First LLM Call
====================================
Goal: Connect to a local Ollama model and get your first response from an LLM.

What you learn:
- How to import and use ChatOllama (local LLM, no API key needed)
- How to send a message to the LLM using .invoke()
- How to extract and print the text response

Prerequisites:
- Install Ollama: https://ollama.com
- Pull the model: ollama pull llama3.2

Run: python chapter-01-setup/hello_world.py
"""

from langchain_ollama import ChatOllama

# Step 1: Create the LLM — points to your locally running Ollama model
# No API key or credentials needed — Ollama runs on your machine
llm = ChatOllama(model="llama3.2")

# Step 2: Send a message using .invoke()
# Pass a plain string — LangChain wraps it in a HumanMessage automatically
response = llm.invoke("Say hello to a new LangChain student")

# Step 3: Extract and print the text from the response
# .content gives you the plain text string from the AIMessage object
print("Model response:", response.content)
