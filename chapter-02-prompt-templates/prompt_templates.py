"""
Chapter 02 — Prompt Templates
==============================
Goal: Use LangChain prompt templates to build reusable, dynamic prompts.

What you learn:
- PromptTemplate: simple text prompts with {variable} placeholders
- ChatPromptTemplate: chat-style prompts with system + human messages
- .format_messages(): fills in placeholders and returns message objects
- How to send a formatted prompt to a local Ollama model via LCEL chain

Run: python chapter-02-prompt-templates/prompt_templates.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use Ollama locally — no API key needed
llm = ChatOllama(model="llama3.2")

# ChatPromptTemplate defines system + human messages as a reusable template
# {topic} is a placeholder — filled in when you call .format_messages() or .invoke()
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher who explains things in simple terms."),
    ("human", "Explain {topic} in simple terms for a complete beginner.")
])

# You can inspect the formatted messages before sending
formatted_messages = chat_template.format_messages(topic="Python decorators")
print("Formatted messages:", formatted_messages)

# Build a simple chain: prompt -> llm -> parser
# StrOutputParser extracts the plain text string from the AIMessage
chain = chat_template | llm | StrOutputParser()

print("===================================================")
response = chain.invoke({"topic": "Python decorators"})
print("Model says:", response)
print("===================================================")
