"""
Chapter 05 — Memory: Basic Test
================================
Goal: Prove that the chatbot remembers previous messages across multiple turns.

What you learn:
- InMemoryChatMessageHistory: stores conversation history in RAM
- RunnableWithMessageHistory: wraps a chain to automatically inject history
- MessagesPlaceholder: reserves a slot in the prompt where history is inserted
- session_id: allows multiple separate conversations to coexist

How it works:
  Turn 1: "My name is Bhargav" -> saved to history
  Turn 2: "What is my name?"   -> history injected into prompt -> LLM knows the name

Run: python chapter-05-memory/chatbot.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="llama3.2")

# store holds chat histories keyed by session_id
# Multiple users/sessions can have completely separate histories
store = {}

def get_session_history(session_id: str):
    """Return existing history for this session, or create a new one."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# MessagesPlaceholder tells LangChain: "insert the full chat history here"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember what the user tells you."),
    MessagesPlaceholder(variable_name="history"),  # <- history injected here
    ("human", "{input}")
])

chain = prompt | llm

# RunnableWithMessageHistory wraps the chain and handles history automatically
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # which key in input dict is the new message
    history_messages_key="history"  # which placeholder to fill with history
)

config = {"configurable": {"session_id": "user1"}}

# Turn 1: introduce yourself
response1 = chatbot.invoke({"input": "Hi, my name is Bhargav"}, config=config)
print("Bot:", response1.content)

# Turn 2: ask something that requires memory of turn 1
response2 = chatbot.invoke({"input": "What is my name?"}, config=config)
print("Bot:", response2.content)
