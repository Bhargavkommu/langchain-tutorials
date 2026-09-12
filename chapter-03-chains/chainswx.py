"""
Chapter 03 — Chains with IBM watsonx (bonus)
=============================================
Goal: Same LCEL chain pattern but using IBM watsonx as the LLM.

What you learn:
- ChatWatsonx: LangChain-compatible wrapper for IBM watsonx models
- You can swap out any LLM in a chain without changing the rest of the code
- This is the power of LCEL — the chain is LLM-agnostic

Run: python chapter-03-chains/chainswx.py
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx

load_dotenv(dotenv_path=".env")
api_key = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")
space_id = os.getenv("WATSONX_SPACE_ID")
deploy_id = os.getenv("WATSONX_DEPLOYMENT_ID")

# ChatWatsonx is a LangChain-native LLM wrapper for IBM watsonx
llm = ChatWatsonx(
    deployment_id=deploy_id,
    url=url,
    api_key=api_key,
    space_id=space_id,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains things simply."),
    ("human", "Explain {topic} in simple terms for a complete beginner.")
])

# Same chain pattern as chains.py — only the LLM changed
chain = prompt | llm
response = chain.invoke({"topic": "Python lists"})
print(response)
