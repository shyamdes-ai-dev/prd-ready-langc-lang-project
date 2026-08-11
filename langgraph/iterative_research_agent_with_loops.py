import operator
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import operator
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def iterative_research():
    """Iterative research that goes deeper based on findings"""

    def research(state: ResearchState) -> dict:
        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
        else:
            query = f"Based on thee findings: \n {state['findings'][-1]}\n\n Go deeper"

        response = model.invoke(query)
        return {
            "findings": [response.content[0].get("text")],
        }

    def generate_questions(state: ResearchState) -> dict:
        response = model.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            "What is one deeper question to explore? Reply with just the question"
        )
        return {
            "questions": [response.content[0].get("text")],
            "iteration": state["iteration"] + 1,
        }

    def synthesize(state: ResearchState) -> dict:
        all_findings = "\n\n".join(state["findings"])
        response = model.invoke(
            f"Synthesize these findings into a coherent summary:\n\n{all_findings}"
        )
        return {"summary": response.content[0].get("text")}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            return "synthesize"
        return "research"

    graph = StateGraph(ResearchState)
    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile()

    print("\n Iterative Research Agent:\n")
    result = app.invoke(
        {
            "topic": "quantum computing applications",
            "findings": [],
            "questions": [],
            "max_depth": 2,
            "summary": "",
            "iteration": 0,
        }
    )

    print(f"Topic: {result['topic']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\n Findings collected: {len(result['findings'])}")
    print(f"\n Final Summary: \n{result['summary']}")


iterative_research()
