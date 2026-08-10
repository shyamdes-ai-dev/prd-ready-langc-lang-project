"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify and resume
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash")

class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str 

def interrupt_for_approval():
    """Interrupt for human approval at critical steps"""

    def create_draft(state: ApprovalState) -> dict:
        response = model.invoke(
            f"Create a professional response for : {state['request']}\n")
        return {'draft': response.content[0].get('text')}

    def wait_for_approval(state:ApprovalState) -> dict:
        
        return state

    def finalize(state:ApprovalState) -> dict:
        if state["approved"]:
            return {"final": state["draft"]}
        else:
            # InCorporate Feedback
            response = model.invoke(
                f"Revise this dreaft based on feedback:\n\n"
                f"Draft: {state['draft']}\n\n"
                f"Feedback: {state['feedback']}"
            )
            return {"final": response.content[0].get("text")}

    graph = StateGraph(ApprovalState)
    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)
    
    #Add checkpointer to enable state saving between interruptions
    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["approval"] # Interrupt before the approval node to allow human intervention
    ) 

    print("Human in the loop Approval Demo\n")

    # Configuration for this thread
    config = {
        "configurable": {
            "thread_id": "demo-1"
        }
    }    

    # Step 1: Run until interrupt    
    result = app.invoke(
        {
        "request": "Write a thank-you email ofr a job interview",
        "draft": "",
        "approved": False,
        "feedback": "",
        "final": ""
    }, config)   

    print(f"\nDraft Created:\n{result['draft'][:200]}....")
    print("\n[Execution paused for human review]")

    # Step 2: Get current state
    current_state = app.get_state(config)
    print(f"\n Current node: {current_state.next}")

    # Step 3: Simulate human feedback and continue
    print("\nStep 2: Human provides feedback and continues....")

    # Update state with human input
    app.update_state(
        config,
        {
            "approved": False, #Request changes
            "feedback": "Make it more concise and add specific mention of the company"
        }
    )
    # Continue execution
    final_result = app.invoke(None, config)
    print(f"Final result:\n {final_result['final']}")


interrupt_for_approval()