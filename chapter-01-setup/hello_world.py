"""
Chapter 01 — Setup & First LLM Call
====================================
Goal: Connect to IBM watsonx and get your first response from an LLM.

What you learn:
- How to load environment variables from a .env file
- How to authenticate with IBM watsonx using Credentials + APIClient
- How to call a deployed LLM model using ModelInference
- How to extract the text response from the result

Run: python chapter-01-setup/hello_world.py
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference

# Load .env file so os.getenv() can read your credentials
load_dotenv(dotenv_path=".env")

# Read credentials from environment variables
api_key = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")
space_id = os.getenv("WATSONX_SPACE_ID")
deploy_id = os.getenv("WATSONX_DEPLOYMENT_ID")

# Step 1: Create credentials object
credentials = Credentials(url=url, api_key=api_key)

# Step 2: Create API client using credentials and your deployment space
client = APIClient(credentials=credentials, space_id=space_id)

# Step 3: Point to your deployed model
model = ModelInference(deployment_id=deploy_id, api_client=client)

# Step 4: Send a message and get a response
# Messages use the same format as OpenAI: a list of {role, content} dicts
response = model.chat(messages=[{"role": "user", "content": "Say hello to a new LangChain student"}])

# Step 5: Extract and print the text from the response
print("Model response:", response["choices"][0]["message"]["content"])
