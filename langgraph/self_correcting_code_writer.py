"""
Cyclic Self-Correcting Code Writer Agent
========================================
Demonstrates an autonomous self-healing agent loop in LangGraph:
1. `generate_code`: Generates initial Python code or revises previous attempts based on syntax errors.
2. `validate_code`: Uses Python's built-in `compile()` AST function to check for SyntaxErrors in sandbox.
3. `should_continue`: If errors occur and `iteration < max_iterations`, routes back to `generate_code`
   with the traceback error message to attempt automated self-correction.

Architecture:
               ┌─────────────────────┐
               │                     ▼
[START] -> [generate_code] ──> [validate_code]
                                     │
                 (Success == True OR Iteration >= Max)
                                     ▼
                                 [finalize] -> [END]
"""

import operator
from typing import Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# Load API credentials
load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class CodeGenState(TypedDict):
    """
    Tracks the coding task, generated source code, error logs, iteration count,
    and success status.
    """
    task: str
    code: str
    errors: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    success: bool


def self_correcting_code():
    """
    Constructs and executes the autonomous self-correcting coding graph.
    """

    # Node 1: Code Generator & Repairer
    def generate_code(state: CodeGenState) -> dict:
        print(f"-> [Node: generate_code] Generating code (Attempt {state['iteration'] + 1})...")
        if state["iteration"] == 0:
            prompt = (
                f"Write pure Python code for: {state['task']}\n"
                f"Return ONLY executable Python code blocks. Do NOT include markdown backticks or explanations."
            )
        else:
            prompt = (
                f"Your previous Python code failed with the following syntax error:\n{state['errors'][-1]}\n\n"
                f"Previous Code:\n{state['code']}\n\n"
                f"Please fix the syntax error and return ONLY the corrected Python code without markdown."
            )

        response = model.invoke(prompt)
        raw_code = response.content[0].get("text") if isinstance(response.content, list) else response.content

        # Clean markdown formatting backticks if present
        clean_code = raw_code.replace("```python", "").replace("```", "").strip()
        return {
            "code": clean_code,
            "iteration": state["iteration"] + 1,
        }

    # Node 2: Syntax Validator
    def validate_code(state: CodeGenState) -> dict:
        print("-> [Node: validate_code] Validating Python AST syntax...")
        code = state["code"]
        try:
            # Check if code compiles into valid bytecode AST
            compile(code, "<string>", "exec")
            print("   Validation Result: SUCCESS (Syntax is valid)")
            return {"success": True}
        except SyntaxError as e:
            error_msg = f"SyntaxError on line {e.lineno}: {e.msg}"
            print(f"   Validation Result: FAILED -> {error_msg}")
            return {"success": False, "errors": [error_msg]}
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"   Validation Result: FAILED -> {error_msg}")
            return {"success": False, "errors": [error_msg]}

    # Node 3: Finalizer
    def finalize(state: CodeGenState) -> dict:
        print("-> [Node: finalize] Finalizing execution.")
        return state

    # Conditional Router Function
    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        # Exit if validation succeeded OR safety iteration limit reached
        if state["success"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        return "generate"

    # Assemble Graph
    workflow = StateGraph(CodeGenState)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("validate_code", validate_code)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "validate_code")
    workflow.add_conditional_edges(
        "validate_code",
        should_continue,
        {
            "generate": "generate_code",
            "end": "finalize",
        },
    )
    workflow.add_edge("finalize", END)

    graph = workflow.compile()

    print("=== Running Self-Correcting Code Agent Loop ===")
    inputs = {
        "task": "Write a function to check if a number is prime and return boolean.",
        "max_iterations": 3,
        "iteration": 0,
        "success": False,
        "code": "",
        "errors": [],
    }

    result = graph.invoke(inputs)

    print("\n" + "=" * 60)
    print("Agent Execution Summary:")
    print(f"Task             : {result['task']}")
    print(f"Validation State : {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Total Iterations : {result['iteration']}")
    print(f"Error History    : {result['errors']}")
    print(f"\nFinal Generated Code:\n{result['code']}")
    print("=" * 60)


if __name__ == "__main__":
    self_correcting_code()
