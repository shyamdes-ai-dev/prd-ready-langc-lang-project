from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import operator
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")

"""

Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement

"""


class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    success: bool


def self_correcting_code():
    """self-correcting code generator."""

    def generate_code(state: CodeGenState) -> dict:
        if state["iteration"] == 0:
            prompt = f"Write Python code for: {state['task']}\nReturn only the code blocks. no explanation and no markdown."
        else:
            prompt = f"Please fix the following errors in the code:\n\n{state['errors']}\n\nOriginal code:\n\n{state['code']}. Only generate the code blocks"
        response = model.invoke(prompt)
        return {
            "code": response.content[0].get("text"),
            "iteration": state["iteration"] + 1,
        }

    def validate_code(state: CodeGenState) -> dict:
        code = state["code"]

        try:
            compile(code, "<string>", "exec")
            return {"success": True}
        except SyntaxError as e:
            return {"errors": [f"SyntaxError: {str(e)}"]}
        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        return "generate"

    def finalize(state: CodeGenState) -> dict:
        return state

    workflow = StateGraph(CodeGenState)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("validate_code", validate_code)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "validate_code")
    workflow.add_conditional_edges(
        "validate_code",
        should_continue,
        {"generate": "generate_code", "end": "finalize"},
    )
    workflow.add_edge("finalize", END)

    graph = workflow.compile()

    inputs = {
        "task": "write a function to check if a number is prime. which has syntax Error",
        "max_iterations": 3,
        "iteration": 0,
        "success": False,
        "code": "",
        "errors": [],
    }

    result = graph.invoke(inputs)
    print("Task: ", result["task"])
    print("=" * 60)
    print("Code: ", result["code"])
    print("=" * 60)
    print("Errors: ", result["errors"])
    print("Iteration: ", result["iteration"])
    print("Success: ", result["success"])


self_correcting_code()
