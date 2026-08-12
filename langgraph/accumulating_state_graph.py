"""
LangGraph State Reducers & Value Streaming
==========================================
Demonstrates how LangGraph handles state updates using Reducers (`operator.add`):
- By default, returning a key from a node *overwrites* the previous value.
- Wrapping a field in `Annotated[T, operator.add]` tells LangGraph to *accumulate*
  (append to lists, add to numeric counters) instead of replacing the value.
- Demonstrates streaming intermediate state values using `.stream(..., stream_mode="values")`.
"""

import operator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# Load environment configuration
load_dotenv()


# -----------------------------------------------------------------------------
# State Schema with Reducers
# -----------------------------------------------------------------------------
class AccumulatingState(TypedDict):
    """
    State definition using Annotated reducers:
    - messages: Uses operator.add to append new messages to the existing list.
    - count   : Uses operator.add to sum incoming integer updates to the running total.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    count: Annotated[int, operator.add]


def accumulating_state():
    """
    Constructs a two-step graph where each node appends messages and increments the counter,
    and streams the evolving state values in real-time.
    """

    # Node 1: Appends a human message and increments count by 1
    def step_one(state: AccumulatingState) -> dict:
        return {
            "messages": [HumanMessage(content="Hello from step one")],
            "count": 1,
        }

    # Node 2: Appends an AI message and increments count by 1
    def step_two(state: AccumulatingState) -> dict:
        return {
            "messages": [AIMessage(content="Hi from step two")],
            "count": 1,
        }

    # Initialize graph with accumulating state schema
    graph = StateGraph(AccumulatingState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    # Wire linear pipeline: START -> step_one -> step_two -> END
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app = graph.compile()

    print("=== Streaming Graph State Updates (stream_mode='values') ===")
    initial_payload = {
        "messages": [HumanMessage(content="Starting conversation message")],
        "count": 0,
    }

    # Stream state snapshots after each node completes
    for step_index, snapshot in enumerate(app.stream(initial_payload, stream_mode="values"), start=1):
        print(f"\n[Snapshot {step_index}]")
        print(f"Total Messages : {len(snapshot['messages'])}")
        print(f"Running Count  : {snapshot['count']}")
        print(f"Latest Message : {snapshot['messages'][-1].content}")


if __name__ == "__main__":
    accumulating_state()
