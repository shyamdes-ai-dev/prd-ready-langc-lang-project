"""
Multi-Node Sequential Agent Pipeline
====================================
Demonstrates a multi-stage sequential agent workflow in LangGraph:
1. `analyze` : Takes raw user input and generates a concise analytical summary.
2. `enhance` : Enriches and expands the analyzed concepts with creative depth.
3. `final`   : Synthesizes the enhanced draft into a polished final response.

Architecture:
[START] -> [Analyze Node] -> [Enhance Node] -> [Final Node] -> [END]
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Load API credentials
load_dotenv()


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class MultiStepState(TypedDict):
    """
    State container tracking each progressive stage of the text generation pipeline.
    """
    input: str
    analyzed: str
    enhanced: str
    final: str


def multi_node_graph():
    """
    Constructs and executes the 3-stage transformation pipeline.
    """
    llm = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")

    # Stage 1: Analyze input
    def analyze_node(state: MultiStepState) -> dict:
        print("-> [Node 1: Analyze] Processing raw input...")
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Analyze the following input and summarize it in one concise sentence.",
                        state["input"],
                    ]
                )
            ]
        )
        analyzed_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        print(f"   Analyzed: {analyzed_text}\n")
        return {"analyzed": analyzed_text}

    # Stage 2: Enhance analysis
    def enhance_node(state: MultiStepState) -> dict:
        print("-> [Node 2: Enhance] Adding depth and creative nuance...")
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Enhance the following input, making it more detailed and engaging.",
                        state["analyzed"],
                    ]
                )
            ]
        )
        enhanced_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        print(f"   Enhanced: {enhanced_text}\n")
        return {"enhanced": enhanced_text}

    # Stage 3: Finalize content
    def final_node(state: MultiStepState) -> dict:
        print("-> [Node 3: Finalize] Polishing final response...")
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Create a final polished output based on this draft without markdown formatting:",
                        state["enhanced"],
                    ]
                )
            ]
        )
        final_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        print(f"   Final: {final_text}\n")
        return {"final": final_text}

    # Build Graph
    graph = StateGraph(MultiStepState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("enhance", enhance_node)
    graph.add_node("final", final_node)

    # Wire Edges
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "enhance")
    graph.add_edge("enhance", "final")
    graph.add_edge("final", END)

    app = graph.compile()

    print("=== Running Multi-Node Sequential Agent ===")
    result = app.invoke(
        {
            "input": "Artificial intelligence is changing the way software developers write code.",
            "analyzed": "",
            "enhanced": "",
            "final": "",
        }
    )

    print("=== Complete Pipeline Summary ===")
    print(f"Initial Input : {result['input']}")
    print(f"Final Output  : {result['final']}")


if __name__ == "__main__":
    multi_node_graph()
