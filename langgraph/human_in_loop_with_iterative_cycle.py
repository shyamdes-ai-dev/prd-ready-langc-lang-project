"""
Multi-Round Human-in-the-Loop Document Review
==============================================
Demonstrates multi-round collaborative human-AI document refinement in LangGraph:
1. AI drafts or updates a document based on human feedback.
2. Execution pauses at an interruption gate (`interrupt_before=["submit"]`).
3. Human reviews the document:
   - If human provides more comments (`needs_revision`): loops back to `apply_feedback` -> `submit`.
   - If human marks `status="approved"`              : routes to `done` -> `END`.

Architecture:
               ┌────────────────────────┐
               │                        ▼
[START] -> [submit] (PAUSE) ──> [apply_feedback]
               │ (status == "approved")
               ▼
            [done] -> [END]
"""

from typing import Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Load API credentials
load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash")


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class ReviewState(TypedDict):
    """
    Tracks document content, history of human review comments, total revision count,
    and workflow status ('pending_review', 'needs_revision', 'approved', 'finalized').
    """
    document: str
    review_comments: list[str]
    revision_count: int
    status: str


def iterative_review():
    """
    Executes a multi-round human document review cycle with state inspection and mutation.
    """

    # Node 1: Submission Gate
    def submit_for_review(state: ReviewState) -> dict:
        print(f"-> [Node: submit] Document submitted for review (Revision {state['revision_count']}).")
        return {"status": "pending_review"}

    # Node 2: Apply Feedback
    def apply_feedback(state: ReviewState) -> dict:
        if not state["review_comments"]:
            return state

        latest_feedback = state["review_comments"][-1]
        print(f"-> [Node: apply_feedback] Revising document based on feedback: '{latest_feedback}'...")

        response = model.invoke(
            f"Revise this document based on the reviewer feedback:\n\n"
            f"Document: {state['document']}\n\n"
            f"Feedback: {latest_feedback}\n\n"
            f"Return ONLY the revised document text."
        )
        revised_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {
            "document": revised_text.strip(),
            "revision_count": state["revision_count"] + 1,
            "status": "revised",
        }

    # Conditional Routing Function after review
    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        if state["status"] == "approved":
            return "done"
        return "apply"

    # Node 3: Finalize
    def finalize(state: ReviewState) -> dict:
        print("-> [Node: done] Document formally approved and finalized.")
        return {"status": "finalized"}

    # Assemble Graph
    graph = StateGraph(ReviewState)
    graph.add_node("submit", submit_for_review)
    graph.add_node("apply_feedback", apply_feedback)
    graph.add_node("done", finalize)

    graph.add_edge(START, "submit")
    graph.add_conditional_edges(
        "submit",
        route_after_review,
        {
            "apply": "apply_feedback",
            "done": "done",
        },
    )
    # Loop back to submit after feedback is applied
    graph.add_edge("apply_feedback", "submit")
    graph.add_edge("done", END)

    # Compile with memory saver checkpointer and pause before submission gate
    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["submit"],
    )

    print("\n=== Running Multi-Round Collaborative Review Demo ===")
    config = {"configurable": {"thread_id": "doc_review_series_01"}}

    # --- Round 1: Initial Submission ---
    print("\n--- [Round 1] Initial AI Submission ---")
    result = app.invoke(
        {
            "document": "Artificial intelligence is a technology that helps computers think and process data.",
            "review_comments": [],
            "revision_count": 0,
            "status": "",
        },
        config,
    )
    print(f"Draft 1: {result['document']}")
    print(">>> [PAUSED: Awaiting Reviewer Comments] <<<")

    # --- Reviewer Round 1 Feedback ---
    print("\n--- [Round 1 Review] Reviewer adds comments requesting technical depth ---")
    app.update_state(
        config,
        {
            "review_comments": ["Add technical depth regarding neural networks and real-world enterprise use cases."],
            "status": "needs_revision",
        },
    )

    # Resume graph: AI applies feedback and reaches 'submit' node again where it pauses
    result2 = app.invoke(None, config)
    print(f"\nDraft 2 (After Revision 1):\n{result2['document'][:180]}...")
    print(">>> [PAUSED: Awaiting Second Review] <<<")

    # --- Reviewer Round 2 Approval ---
    print("\n--- [Round 2 Review] Reviewer approves revised document ---")
    app.update_state(config, {"status": "approved"})

    # Final resume: routes to done
    final_result = app.invoke(None, config)

    print("\n" + "=" * 60)
    print("Final Approved Document:\n")
    print(final_result["document"])
    print("=" * 60)
    print(f"Final Status     : {final_result['status']}")
    print(f"Total Revisions  : {final_result['revision_count']}")


if __name__ == "__main__":
    iterative_review()
