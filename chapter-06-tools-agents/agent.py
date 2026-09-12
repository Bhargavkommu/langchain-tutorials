"""
Chapter 06 — Tools & Agents (ReAct)
=====================================
Goal: Give the LLM "hands" — the ability to call Python functions as tools.

What you learn:
- @tool decorator: turns a Python function into a tool the LLM can call
- Docstring matters: the LLM reads it to decide WHEN to use this tool
- create_react_agent: builds a ReAct agent (Reasoning + Acting loop)
- The agent picks the right tool automatically based on the question
- response["messages"][-1]: the final answer is always the last message

ReAct loop:
  User question
    -> Agent thinks: "I need to multiply"
    -> Agent calls: multiply(6, 7)
    -> Gets result: 42
    -> Agent answers: "The result is 42"

Run: python chapter-06-tools-agents/agent.py
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatOllama(model="llama3.2")

# @tool turns this Python function into a tool the LLM can call
# The docstring is critical — the LLM reads it to know what this tool does
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Create the agent with both tools — it will pick the right one based on the question
agent = create_react_agent(llm, tools=[multiply, add])

# The agent will reason: "plus = addition" -> call add(15, 27) -> return 42
response = agent.invoke({
    "messages": [("human", "What is 15 plus 27?")]
})

# The agent produces multiple internal messages (thinking, tool call, tool result)
# [-1] always gets the last message — the final answer
print(response["messages"][-1].content)
