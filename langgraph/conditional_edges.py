"""
Dynamic Conditional Routing & Intent Dispatch
=============================================
Demonstrates how to route graph execution dynamically based on runtime classification:
1. `classify`: Node that inspects the query and predicts its intent ('question', 'command', or 'statement').
2. `route_by_type`: Conditional routing function that reads the classified state and returns the next node target.
3. Specialized Handlers: Distinct nodes (`handle_question`, `handle_command`, `handle_statement`)
   tailored to process each specific intent.

Architecture:
               ┌─> [handle_question] ─┐
[START] -> [classify] ──> [handle_command]  ──┴─> [END]
               └─> [handle_statement]─┘
"""

from typing import Literal
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
class RouterState(TypedDict):
    """
    Tracks the user's input query, the classified intent type, and the final response.
    """
    query: str
    query_type: str
    response: str


def router():
    """
    Builds and tests the conditional intent-based routing graph.
    """

    # Classifier Node: determines intent type
    def classify_query(state: RouterState) -> dict:
        response = model.invoke(
            f"Classify this query strictly as 'question', 'command', or 'statement'. "
            f"Reply with just that single word.\n\nQuery: {state['query']}"
        )
        text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        classified_type = text.lower().strip()
        print(f"[Classifier] Query: '{state['query']}' -> Intent: '{classified_type}'")
        return {"query_type": classified_type}

    # Specialized Handler Node 1: Questions
    def handle_question(state: RouterState) -> dict:
        response = model.invoke(f"Answer this question concisely: {state['query']}")
        text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {"response": f"[Answer] {text}"}

    # Specialized Handler Node 2: Commands
    def handle_command(state: RouterState) -> dict:
        return {"response": f"[Executing Command] Acknowledged task: '{state['query']}'. Scheduled for execution."}

    # Specialized Handler Node 3: Statements
    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged Statement] Thank you for sharing: '{state['query']}'."}

    # Conditional Routing Function: maps classified state to target node names
    def route_by_type(state: RouterState) -> Literal["handle_question", "handle_command", "handle_statement"]:
        qt = state["query_type"]
        if "question" in qt:
            return "handle_question"
        elif "command" in qt:
            return "handle_command"
        else:
            return "handle_statement"

    # Assemble Graph
    workflow = StateGraph(RouterState)

    # Register Nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("handle_question", handle_question)
    workflow.add_node("handle_command", handle_command)
    workflow.add_node("handle_statement", handle_statement)

    # Add Edges
    workflow.add_edge(START, "classify")

    # Add Conditional Routing Edge from 'classify'
    workflow.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "handle_question": "handle_question",
            "handle_command": "handle_command",
            "handle_statement": "handle_statement",
        },
    )

    # Connect all handlers to END
    workflow.add_edge("handle_question", END)
    workflow.add_edge("handle_command", END)
    workflow.add_edge("handle_statement", END)

    graph = workflow.compile()

    # Test diverse query types
    queries = [
        "Send an urgent status email to the project manager",
        "I really enjoy learning LangGraph and building AI agents",
        "Who is the current Prime Minister of India?",
    ]

    print("\n=== Running Intent Routing Demonstration ===")
    for q in queries:
        result = graph.invoke({"query": q, "query_type": "", "response": ""})
        print(f"Query    : {result['query']}")
        print(f"Type     : {result['query_type']}")
        print(f"Response : {result['response']}")
        print("-" * 60)


if __name__ == "__main__":
    router()
