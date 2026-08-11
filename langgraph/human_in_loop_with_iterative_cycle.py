from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash")


class ReviewState(TypedDict):
    document: str
    review_comments: list[str]
    revision_count: int
    status: str


def iterative_review():
    """Multiple Rounds of Human Review"""

    def submit_for_review(state: ReviewState) -> dict:
        return {"status": "pending_review"}

    def apply_feedback(state: ReviewState) -> dict:
        if not state["review_comments"]:
            return state

        feedback = state["review_comments"][-1]
        response = model.invoke(f"""
            Revise this document based on feedback\n\n:
            Document: {state["document"]}\n\n
            Feedback: {feedback}\n\n
            Return ONLY the revised document, without any other text.
            """)
        return {
            "document": response.content[0].get("text"),
            "revision_count": state["revision_count"] + 1,
            "status": "revised",
        }

    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        if state["status"] == "approved":
            return "done"
        return "apply"

    def finalize(state: ReviewState) -> dict:
        return {"status": "finalized"}

    graph = StateGraph(ReviewState)

    graph.add_node("submit", submit_for_review)
    graph.add_node("apply_feedback", apply_feedback)
    graph.add_node("done", finalize)

    graph.add_edge(START, "submit")

    graph.add_conditional_edges(
        "submit", route_after_review, {"apply": "apply_feedback", "done": "done"}
    )

    graph.add_edge("apply_feedback", "submit")
    graph.add_edge("done", END)

    app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["submit"])

    print("\n Iterative Review Demo:\n")
    print("=" * 60)

    config = {"configurable": {"thread_id": "review-1"}}

    # Step 1: Initial submission - AI creates draft and waits for review
    print("[Step 1] Initial Submission:")
    print("=" * 60)

    result = app.invoke(
        {
            "document": "AI is technology that helps computers think.",
            "review_comments": [],
            "revision_count": 0,
            "status": "",
        },
        config,
    )

    print(f"Initial document: {result['document']}")
    print("\n[Execution paused for human review]")

    # Simulate reviewer adding comments
    app.update_state(
        config,
        {
            "review_comments": ["Add more technical depth and examples"],
            "status": "needs_revision",
        },
    )

    result = app.invoke(None, config)
    print(f"\nAfter revision 1: {result['document'][:150]}......")

    # Final Approval
    app.update_state(config, {"status": "approved"})

    final_result = app.invoke(None, config)
    print(f"\nAfter final approval: {final_result['document'][:150]}......")
    print("-" * 60)
    print(f"\nFinal status: {final_result['status']}")
    print(f"\nTotal revisions: {final_result['revision_count']}")


iterative_review()
