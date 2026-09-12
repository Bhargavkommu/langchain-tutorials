"""
Chapter 02 — Prompt Templates
==============================
Goal: Use LangChain prompt templates to build reusable, dynamic prompts.

What you learn:
- PromptTemplate: simple text prompts with {variable} placeholders
- ChatPromptTemplate: chat-style prompts with system + human messages
- .format_messages(): fills in placeholders and returns message objects
- How to send a formatted prompt to a watsonx model

Run: python chapter-02-prompt-templates/prompt_templates.py
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.prompts import ChatPromptTemplate

# Load credentials from .env
load_dotenv(dotenv_path=".env")
api_key = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")
space_id = os.getenv("WATSONX_SPACE_ID")
deploy_id = os.getenv("WATSONX_DEPLOYMENT_ID")

credentials = Credentials(url=url, api_key=api_key)
client = APIClient(credentials=credentials, space_id=space_id)
model = ModelInference(deployment_id=deploy_id, api_client=client)

# ChatPromptTemplate defines system + human messages as a reusable template
# {topic} is a placeholder — filled in when you call .format_messages()
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher who explains things in simple terms."),
    ("human", "Explain {topic} in simple terms for a complete beginner.")
])

# Fill in the placeholder — returns a list of message objects
formatted_messages = chat_template.format_messages(topic="Python decorators")
print("Formatted messages:", formatted_messages)

# Extract the human message text and send to watsonx
messages = [{"role": "user", "content": formatted_messages[1].content}]
response = model.chat(messages=messages)
print("===================================================")
print("Model says:", response["choices"][0]["message"]["content"])
print("===================================================")
