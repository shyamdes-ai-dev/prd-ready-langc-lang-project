from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class QualityState(TypedDict):
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int


def conditional_looping_graph() -> dict:
    """
    Create a conditional loop graph that evaluates the quality of the content and returns the feedback and quality score.
    """

    def evaluate_quality(state: QualityState) -> dict:
        response = model.invoke(
            f"Rate this content quality from 1-10. Reply with just the number.\n\n"
            f"Content: {state['content']}"
        )
        try:
            score = int(response.content[0].get("text"))
        except:
            score = 5
        return {"quality_score": score}

    def improve_content(state: QualityState):
        response = model.invoke(
            f"Improve this content based on the feedback. Reply with just the improved content.\n\n"
            f"Content: {state['content']}\n"
        )
        return {
            "content": response.content[0].get("text"),
            "iteration": state["iteration"] + 1,
        }

    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state["content"],
            "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}",
        }

    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 7 or state["iteration"] >= 2:
            return "finalize"
        return "improve"

    workflow = StateGraph(QualityState)
    workflow.add_node("evaluate_quality", evaluate_quality)
    workflow.add_node("improve_content", improve_content)
    workflow.add_node("finalize_content", finalize_content)

    workflow.add_edge(START, "evaluate_quality")
    workflow.add_conditional_edges(
        "evaluate_quality",
        should_continue,
        {"improve": "improve_content", "finalize": "finalize_content"},
    )
    workflow.add_edge("improve_content", "evaluate_quality")
    workflow.add_edge("finalize_content", END)

    graph = workflow.compile()

    result = graph.invoke(
        {
            "content": "Write a short story about a robot who discovered emotions.",
            "iteration": 0,
        }
    )

    print("Final Content: ", result["final_content"])
    print("Feedback: ", result["feedback"])
    print("Iteration: ", result["iteration"])
    print("Quality Score: ", result["quality_score"])


conditional_looping_graph()
