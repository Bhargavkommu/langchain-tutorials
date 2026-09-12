"""
Chapter 10 — LangGraph Introduction
=====================================
Goal: Build your first LangGraph — a stateful, graph-based AI workflow.

What you learn:
- StateGraph: the graph container you define your workflow in
- TypedDict / State: a shared dictionary passed between all nodes
- Node: a Python function that reads from state and writes back to state
- Edge: a directed connection between two nodes (defines the execution flow)
- START / END: built-in entry and exit points of the graph
- graph.compile(): validates and locks the graph before running
- app.invoke(): runs the compiled graph with an initial state

Graph flow in this file:
  START -> answer_question -> END

Key rule for nodes:
  Every node receives the FULL state as input.
  Every node returns ONLY the fields it wants to update (not the whole state).

Why LangGraph over a simple chain?
  Chains:    A -> B -> C  (linear only)
  LangGraph: supports branching, looping, conditional routing
             essential for building real agentic AI systems

Run: python chapter-10-langgraph/graph_app.py
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# State = the shared memory passed between all nodes
# TypedDict enforces the shape — each field has a name and type
class State(TypedDict):
    question: str   # input: the user's question
    answer: str     # output: the LLM's response


llm = ChatOllama(model="llama3.2")


def answer_question(state: State) -> dict:
    """
    Node: reads the question from state, calls the LLM, returns the answer.
    Returns only the fields it updates — LangGraph merges this into the full state.
    """
    response = llm.invoke(state["question"])
    return {"answer": response.content}


# Build the graph
graph = StateGraph(State)

# Register the node with a string name
graph.add_node("answer_question", answer_question)

# Define the execution flow
graph.add_edge(START, "answer_question")   # entry point
graph.add_edge("answer_question", END)     # exit point

# Compile: validates the graph (checks all nodes are connected) and makes it runnable
app = graph.compile()

# Run the graph — pass in the initial state
result = app.invoke({"question": "What is Python programming language in one sentence?"})

# Read the final answer from the output state
print(result["answer"])
