"""
Multi-Path 2D Decision Matrix Routing
=====================================
Demonstrates multi-variable dynamic routing in LangGraph based on a 2x2 matrix:
- Variable 1: Urgency   ('urgent' vs 'normal')
- Variable 2: Complexity ('complex' vs 'simple')

Routing Matrix:
┌─────────────────┬───────────────────────────────┬───────────────────────────────┐
│                 │ Simple                        │ Complex                       │
├─────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Urgent          │ -> urgent_simple_handler      │ -> urgent_complex_handler     │
│                 │    (Quick Response Team)      │    (Senior Escalation Team)   │
├─────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Normal          │ -> normal_simple_handler      │ -> normal_complex_handler     │
│                 │    (Standard Support Tier)    │    (Specialized Team)         │
└─────────────────┴───────────────────────────────┴───────────────────────────────┘
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Load API credentials
load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class TaskState(TypedDict):
    """
    Tracks the input task, evaluated dimensions (urgency, complexity), the assigned handler,
    and the final resolution.
    """
    task: str
    urgency: str
    complexity: str
    handler: str
    result: str


def multi_path_routing():
    """
    Constructs and executes the 2D matrix decision routing graph.
    """

    # Analysis Node: Evaluates urgency and complexity along two distinct axes
    def analyze_task(state: TaskState) -> dict:
        urgency_response = model.invoke(
            f"Is this task urgent? Reply strictly 'urgent' or 'normal'.\nTask: {state['task']}"
        )
        urgency_text = urgency_response.content[0].get("text") if isinstance(urgency_response.content, list) else urgency_response.content

        complexity_response = model.invoke(
            f"Is this task technically complex? Reply strictly 'complex' or 'simple'.\nTask: {state['task']}"
        )
        complexity_text = complexity_response.content[0].get("text") if isinstance(complexity_response.content, list) else complexity_response.content

        return {
            "urgency": urgency_text.lower().strip(),
            "complexity": complexity_text.lower().strip(),
        }

    # Handler Tier 1: Urgent & Complex
    def urgent_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Senior Engineering Team",
            "result": "Escalated immediately to Senior Staff on-call for high-priority crisis intervention.",
        }

    # Handler Tier 2: Urgent & Simple
    def urgent_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Quick Response Team",
            "result": "Dispatched to automated fast-track queue for immediate resolution.",
        }

    # Handler Tier 3: Normal & Complex
    def normal_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Specialized Architecture Team",
            "result": "Assigned to specialized architects for scheduled deep-dive review.",
        }

    # Handler Tier 4: Normal & Simple
    def normal_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Standard Support Queue",
            "result": "Routed through standard operational workflows.",
        }

    # Matrix Router Predicate Function
    def route_task(state: TaskState) -> str:
        is_urgent = "urgent" in state["urgency"]
        is_complex = "complex" in state["complexity"]

        if is_urgent:
            return "urgent_complex_handler" if is_complex else "urgent_simple_handler"
        else:
            return "normal_complex_handler" if is_complex else "normal_simple_handler"

    # Finalizer Node: Combines handler results into final output
    def finalize_task(state: TaskState) -> dict:
        return {"result": f"[{state['handler']}] {state['result']}"}

    # Assemble Graph
    workflow = StateGraph(TaskState)

    # Register Nodes
    workflow.add_node("analyze_task", analyze_task)
    workflow.add_node("urgent_complex_handler", urgent_complex_handler)
    workflow.add_node("urgent_simple_handler", urgent_simple_handler)
    workflow.add_node("normal_complex_handler", normal_complex_handler)
    workflow.add_node("normal_simple_handler", normal_simple_handler)
    workflow.add_node("finalize_task", finalize_task)

    # Add Edges
    workflow.add_edge(START, "analyze_task")
    workflow.add_conditional_edges(
        "analyze_task",
        route_task,
        {
            "urgent_complex_handler": "urgent_complex_handler",
            "urgent_simple_handler": "urgent_simple_handler",
            "normal_complex_handler": "normal_complex_handler",
            "normal_simple_handler": "normal_simple_handler",
        },
    )
    workflow.add_edge("urgent_complex_handler", "finalize_task")
    workflow.add_edge("urgent_simple_handler", "finalize_task")
    workflow.add_edge("normal_complex_handler", "finalize_task")
    workflow.add_edge("normal_simple_handler", "finalize_task")
    workflow.add_edge("finalize_task", END)

    graph = workflow.compile()

    # Test diverse task scenarios across the matrix
    tasks = [
        "Production Database is down! Customers experiencing 500 errors!",
        "Fix a minor typo in the website footer copyright notice.",
        "Redesign the entire distributed multi-region database architecture.",
        "Update the API documentation for the billing endpoint.",
    ]

    print("\n=== Running 2D Matrix Task Routing Demonstration ===")
    for t in tasks:
        result = graph.invoke(
            {
                "task": t,
                "urgency": "",
                "complexity": "",
                "handler": "",
                "result": "",
            }
        )
        print(f"Task       : {result['task']}")
        print(f"Dimensions : Urgency='{result['urgency']}' | Complexity='{result['complexity']}'")
        print(f"Assigned   : {result['handler']}")
        print(f"Resolution : {result['result']}")
        print("-" * 60)


if __name__ == "__main__":
    multi_path_routing()
