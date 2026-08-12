"""
LangGraph Core Fundamentals
===========================
Demonstrates the foundational components of LangGraph:
1. State Schema Definition using Python `TypedDict`.
2. Node Function definitions that accept state and return state updates.
3. Graph Construction (`StateGraph`), adding nodes and explicit edges (`START` -> Node -> `END`).
4. Graph Compilation (`graph.compile()`).
5. Graph Visualization using Mermaid PNG export (`draw_mermaid_png`).
6. Graph Invocation with initial state dictionary.
"""

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Load environment configuration
load_dotenv()


# -----------------------------------------------------------------------------
# 1. State Schema Definition
# -----------------------------------------------------------------------------
class SimpleState(TypedDict):
    """
    TypedDict defining the state container for the graph.
    Every node in the graph reads from and writes to this structure.
    """
    input: str
    output: str
    step: int


# -----------------------------------------------------------------------------
# 2. Graph Construction & Execution
# -----------------------------------------------------------------------------
def simple_graph():
    """
    Constructs, compiles, visualizes, and executes a minimal single-node graph.
    """

    # Node function: processes state and returns a dictionary of updated fields
    def process(state: SimpleState) -> dict:
        return {
            "output": state["input"].upper(),
            "step": state["step"] + 1,
        }

    # Step 1: Initialize StateGraph with the schema
    graph = StateGraph(SimpleState)

    # Step 2: Register nodes (callable functions)
    graph.add_node("process", process)

    # Step 3: Define edges connecting START to node, and node to END
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # Step 4: Compile the graph into an executable Runnable
    workflow = graph.compile()

    # Step 5: Render and export the graph architecture diagram to PNG
    try:
        png_bytes = workflow.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graph visualization successfully saved to graph.png")
    except Exception as e:
        print("Could not generate graph.png diagram:", e)

    # Step 6: Invoke the compiled workflow with initial state inputs
    initial_input = {"input": "hello world", "step": 0, "output": ""}
    result = workflow.invoke(initial_input)

    print("\n--- Graph Execution Result ---")
    print(f"Input : {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Step  : {result['step']}")


if __name__ == "__main__":
    simple_graph()
