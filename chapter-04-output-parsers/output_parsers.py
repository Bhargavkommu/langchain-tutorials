"""
Chapter 04 — Output Parsers: JSON
==================================
Goal: Make the LLM return structured data (a Python dict) instead of plain text.

What you learn:
- JsonOutputParser: instructs the LLM to return valid JSON, then parses it into a dict
- Difference between str (plain text) vs dict (structured data)
- Accessing dict values with result["key"]

Why output parsers?
  By default LLMs return plain text strings.
  Output parsers let you get back structured data you can actually use in code.

Run: python chapter-04-output-parsers/output_parsers.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model="llama3.2")

# JsonOutputParser converts the LLM's JSON string output into a Python dict
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always respond with valid JSON only. No extra text."),
    ("human", "Create a fictional {profession} profile with fields: name, age, and hobby. Return as JSON.")
])

# chain: prompt -> LLM -> parser (converts JSON string to Python dict)
chain = prompt | llm | parser

result = chain.invoke({"profession": "chef"})
print(result)              # prints the full dict
print(type(result))        # <class 'dict'>
print("Name:", result["name"])
print("Age:", result["age"])
print("Hobby:", result["hobby"])
