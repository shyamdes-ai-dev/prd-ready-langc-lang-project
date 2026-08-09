from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class TaskState(TypedDict):
    task: str
    urgency: str
    complexity: str
    handler: str
    result: str


def multi_path_routing():
    def analyze_task(state: TaskState) -> dict:
        # Analyze the task and determine the urgency and complexity
        urgency_response = model.invoke(
            f"Is this task urgent? Reply 'urgent' or 'normal'. \nTask: {state['task']}"
        )
        urgency = urgency_response.content[0].get("text")

        complexity_response = model.invoke(
            f" Is this task complex? Reply 'complex' or 'simple'.\n Task: {state['task']}"
        )
        complexity = complexity_response.content[0].get("text")

        return {
            "urgency": urgency.lower().strip(),
            "complexity": complexity.lower().strip(),
        }

    def urgent_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Senior Team",
            "result": "Escalated to senior team for immediate action",
        }

    def urgent_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Quick Response Team",
            "result": "Handled immediately by available agent",
        }

    def normal_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Specialized Team",
            "result": "Assigned to specialized team for expert handling",
        }

    def normal_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Standard Support",
            "result": "Handled through standard support channel",
        }

    def route_task(state: TaskState) -> str:
        is_urgent = "urgent" in state["urgency"]
        is_complex = "complex" in state["complexity"]

        if is_urgent:
            if is_complex:
                return "urgent_complex_handler"
            else:
                return "urgent_simple_handler"
        else:
            if is_complex:
                return "normal_complex_handler"
            else:
                return "normal_simple_handler"

    def finalize_task(state: TaskState) -> dict:
        return {"result": f"Task completed by {state['handler']}: {state['result']}"}

    workflow = StateGraph(TaskState)
    workflow.add_node("analyze_task", analyze_task)
    workflow.add_node("urgent_complex_handler", urgent_complex_handler)
    workflow.add_node("urgent_simple_handler", urgent_simple_handler)
    workflow.add_node("normal_complex_handler", normal_complex_handler)
    workflow.add_node("normal_simple_handler", normal_simple_handler)
    workflow.add_node("finalize_task", finalize_task)

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

    tasks = [
        "Server is down! Need immediate fix!",
        "Update the documentation for the API",
        "Redesign the entire database schema",
        "Fix the typo on the homepage",
    ]
    for task in tasks:
        result = graph.invoke({"task": task})
        print(f"Task: {task}")
        print(f"Urgency: {result['urgency']} | Complexity: {result['complexity']}")
        print(f"Handler: {result['handler']}")
        print(f"Result: {result['result']}")
        print("-" * 60)


multi_path_routing()
