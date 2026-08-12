"""
Quality Evaluation & Iterative Refinement Loop
==============================================
Demonstrates a cyclic self-refinement agent loop in LangGraph:
1. `evaluate_quality`: Scores the current content draft on a scale of 1-10.
2. `should_continue` : Conditional edge evaluating stopping criteria:
   - If `quality_score >= 7` OR `iteration >= 2`: Route to `finalize_content` -> `END`.
   - Otherwise: Route to `improve_content` -> Loop back to `evaluate_quality`.
3. `improve_content` : Prompts the LLM to enhance the draft based on evaluation.

Architecture:
               ┌───────────────────────┐
               │                       ▼
[START] -> [evaluate_quality] ──> [improve_content]
               │ (Score >= 7 or Iterations >= 2)
               ▼
       [finalize_content] -> [END]
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
class QualityState(TypedDict):
    """
    Tracks the draft content, numeric quality score, feedback summary, and iteration counter.
    """
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int


def conditional_looping_graph():
    """
    Constructs, compiles, and executes the quality-driven refinement loop.
    """

    # Evaluator Node: Grades the current content
    def evaluate_quality(state: QualityState) -> dict:
        print(f"-> [Evaluating Quality] (Iteration {state['iteration']})...")
        response = model.invoke(
            f"Rate the following story quality on an integer scale from 1 to 10. "
            f"Reply with strictly just the integer number.\n\nContent: {state['content']}"
        )
        text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        try:
            score = int(text.strip())
        except Exception:
            score = 5
        print(f"   Quality Score Assigned: {score}/10")
        return {"quality_score": score}

    # Refinement Node: Improves the content draft
    def improve_content(state: QualityState) -> dict:
        print("-> [Improving Content] Generating revised version...")
        response = model.invoke(
            f"Improve this content to make it more emotionally resonant and descriptive. "
            f"Reply with just the revised text.\n\nContent: {state['content']}"
        )
        improved_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {
            "content": improved_text,
            "iteration": state["iteration"] + 1,
        }

    # Finalizer Node: Prepares finalized result and metadata
    def finalize_content(state: QualityState) -> dict:
        print("-> [Finalizing] Quality threshold reached or max iterations met.")
        return {
            "final_content": state["content"],
            "feedback": f"Approved after {state['iteration']} iteration(s) with final score {state['quality_score']}/10.",
        }

    # Conditional Exit Function
    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        # Safe looping guardrails: Score >= 7 OR max iterations >= 2
        if state["quality_score"] >= 7 or state["iteration"] >= 2:
            return "finalize"
        return "improve"

    # Assemble Graph
    workflow = StateGraph(QualityState)
    workflow.add_node("evaluate_quality", evaluate_quality)
    workflow.add_node("improve_content", improve_content)
    workflow.add_node("finalize_content", finalize_content)

    workflow.add_edge(START, "evaluate_quality")
    workflow.add_conditional_edges(
        "evaluate_quality",
        should_continue,
        {
            "improve": "improve_content",
            "finalize": "finalize_content",
        },
    )
    # Loop back edge: from improve_content back to evaluate_quality
    workflow.add_edge("improve_content", "evaluate_quality")
    workflow.add_edge("finalize_content", END)

    graph = workflow.compile()

    print("=== Running Iterative Quality Refinement Loop ===")
    initial_input = {
        "content": "Write a short story about a robot who discovered emotions.",
        "quality_score": 0,
        "feedback": "",
        "final_content": "",
        "iteration": 0,
    }

    result = graph.invoke(initial_input)

    print("\n=== Refinement Complete ===")
    print(f"Total Iterations : {result['iteration']}")
    print(f"Final Score      : {result['quality_score']}/10")
    print(f"Status / Feedback: {result['feedback']}")
    print(f"Final Content    :\n{result['final_content']}")


if __name__ == "__main__":
    conditional_looping_graph()
