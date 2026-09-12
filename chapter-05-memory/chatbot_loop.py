"""
Chapter 05 — Memory: Interactive Chatbot Loop
==============================================
Goal: Build a terminal chatbot that keeps memory across the whole conversation.

What you learn:
- while True loop: keeps the chatbot running until the user types 'quit'
- input(): reads text typed by the user in the terminal
- The chatbot remembers everything said earlier in the session
- Combining memory (Chapter 05) with interactive input

Try it:
  You: My favourite color is blue
  You: What is my favourite color?
  -> The bot will remember!

Run: python chapter-05-memory/chatbot_loop.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="llama3.2")

store = {}

def get_session_history(session_id: str):
    """Return or create a chat history for the given session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember what the user tells you."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "user1"}}

print("Chatbot is ready! Type 'quit' to exit.\n")

# Keep the conversation going until the user types 'quit'
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = chatbot.invoke({"input": user_input}, config=config)
    print("Bot:", response.content)
