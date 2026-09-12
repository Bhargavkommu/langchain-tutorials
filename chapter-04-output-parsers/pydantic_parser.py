"""
Chapter 04 — Output Parsers: Pydantic
======================================
Goal: Make the LLM return a typed Python object using a Pydantic model.

What you learn:
- PydanticOutputParser: parses LLM output into a validated Python object
- BaseModel: defines the exact shape and types of your expected output
- Field: adds a description to each field to guide the LLM
- Dot notation: access fields with result.name instead of result["name"]

Difference from JsonOutputParser:
  JsonOutputParser  -> returns a plain dict  -> result["name"]
  PydanticOutputParser -> returns a typed object -> result.name (with type validation)

Run: python chapter-04-output-parsers/pydantic_parser.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Define the exact shape of the output you expect from the LLM
class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age as a whole number")
    hobby: str = Field(description="The person's main hobby")


llm = ChatOllama(model="llama3.2")

# PydanticOutputParser validates output against the Person schema
parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", 'You are a helpful assistant. Return only a flat JSON object. Example: {{"name": "John", "age": 30, "hobby": "reading"}}'),
    ("human", "Create a fictional {profession}. Return JSON with keys: name (string), age (integer), hobby (string).")
])

chain = prompt | llm | parser

result = chain.invoke({"profession": "scientist"})
print(result)           # Person(name='...', age=..., hobby='...')
print(type(result))     # <class '__main__.Person'>
print(result.name)      # dot notation — type-safe access
print(result.age)
print(result.hobby)
