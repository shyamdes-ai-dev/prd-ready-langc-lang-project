from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import operator
from dotenv import load_dotenv

load_dotenv()


class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str


def multi_node_graph():
    llm = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")

    def analyze_node(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Analyze the following input and summarize it in one sentence.",
                        state["input"],
                    ]
                )
            ]
        )
        print("Analyzed Content: ", response.content[0].get("text"))
        return {"analyzed": response.content[0].get("text")}

    def enhance_node(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Enhance the following input and make it more interesting.",
                        state["analyzed"],
                    ]
                )
            ]
        )
        print("Enhanced Content: ", response.content[0].get("text"))
        return {"enhanced": response.content[0].get("text")}

    def final_node(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=[
                        "Create a final output based on the following input.",
                        state["enhanced"],
                    ]
                )
            ]
        )
        print("Final Content: ", response.content[0].get("text"))
        return {"final": response.content[0].get("text")}

    graph = StateGraph(MultiStepState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("enhance", enhance_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "enhance")
    graph.add_edge("enhance", "final")
    graph.add_edge("final", END)

    app = graph.compile()
    result = app.invoke(
        {
            "input": "Hello, how are you? All reponses should not be in the markdwon language"
        }
    )

    print(f"Final: {result['final']}")


multi_node_graph()
