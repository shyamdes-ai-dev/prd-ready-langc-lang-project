"""
Human-in-the-Loop (HITL) Approval Pattern
=========================================
Demonstrates how to pause graph execution, inspect intermediate state, allow a human
to provide feedback or modify the state, and resume execution cleanly.

Key LangGraph HITL Primitives:
1. `interrupt_before=["approval"]`: Pauses graph execution right before entering the specified node.
2. `app.get_state(config)`       : Inspects current state and identifies next pending node.
3. `app.update_state(config, ...)`: Modifies state variables (e.g. injecting human approval/feedback).
4. `app.invoke(None, config)`     : Resumes execution from the paused checkpoint.
"""

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
class ApprovalState(TypedDict):
    """
    Tracks the generation request, draft content, human approval flag,
    human feedback, and final approved output.
    """
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str


def interrupt_for_approval():
    """
    Executes a workflow where an AI generates a draft, execution pauses for human review,
    and resumes to either finalize or incorporate revisions.
    """

    # Node 1: AI generates initial draft
    def create_draft(state: ApprovalState) -> dict:
        print("-> [Node: draft] Generating initial response draft...")
        response = model.invoke(
            f"Create a professional, polite response for: {state['request']}\n"
        )
        draft_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {"draft": draft_text}

    # Node 2: Approval Gate (Execution pauses before this node)
    def wait_for_approval(state: ApprovalState) -> dict:
        print("-> [Node: approval] Resumed into approval gate...")
        return state

    # Node 3: Finalizer / Revision Handler
    def finalize(state: ApprovalState) -> dict:
        print("-> [Node: finalize] Finalizing content based on human decision...")
        if state["approved"]:
            print("   Status: Approved without changes.")
            return {"final": state["draft"]}
        else:
            print("   Status: Changes requested. Revising with human feedback...")
            response = model.invoke(
                f"Revise this draft incorporating the following feedback:\n\n"
                f"Draft:\n{state['draft']}\n\n"
                f"Feedback:\n{state['feedback']}\n\n"
                f"Return only the revised text."
            )
            revised_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
            return {"final": revised_text}

    # Build Graph
    graph = StateGraph(ApprovalState)
    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)

    # Compile graph with MemorySaver and pause before 'approval' node
    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["approval"],  # Interruption trigger
    )

    print("=== Running Human-in-the-Loop Approval Demo ===\n")
    config = {"configurable": {"thread_id": "interview_email_001"}}

    # Step 1: Run graph until the interrupt is triggered
    print("[Step 1] Initial Invocation (Runs until interrupt):")
    result = app.invoke(
        {
            "request": "Write a thank-you email after a software engineer job interview",
            "draft": "",
            "approved": False,
            "feedback": "",
            "final": "",
        },
        config,
    )

    print(f"\nDraft Created by AI:\n{result['draft'][:220]}...\n")
    print(">>> [EXECUTION PAUSED: Waiting for human intervention] <<<")

    # Step 2: Inspect graph state while paused
    current_state = app.get_state(config)
    print(f"Current Next Node in Queue: {current_state.next}")

    # Step 3: Human reviews the draft, rejects it, and provides specific feedback
    print("\n[Step 2] Human Reviewer rejects draft and inputs feedback:")
    app.update_state(
        config,
        {
            "approved": False,
            "feedback": "Make it shorter, more enthusiastic, and specifically mention excitement about their AI architecture.",
        },
    )

    # Step 4: Resume execution by invoking with None and the same thread config
    print("\n[Step 3] Resuming graph execution with human updates...")
    final_result = app.invoke(None, config)

    print("\n=== Final Output After Human Revision ===")
    print(final_result["final"])


if __name__ == "__main__":
    interrupt_for_approval()
