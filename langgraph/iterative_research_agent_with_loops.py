"""
Autonomous Depth-First Iterative Research Agent
===============================================
Demonstrates an exploratory multi-step research agent that loops to investigate
a topic in increasing depth:
1. `research`: Gathers initial facts on the topic or investigates the latest follow-up question.
2. `generate_questions`: Formulates an inquisitive, deeper follow-up question based on the latest findings.
3. `should_continue`: Routes back to `research` until `max_depth` iterations are satisfied.
4. `synthesize`: Aggregates all collected findings into an executive research briefing.

Architecture:
               ┌──────────────────────┐
               │                      ▼
[START] -> [research] ──> [generate_questions]
                                │
                      (Iteration >= max_depth)
                                ▼
                         [synthesize] -> [END]
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
class ResearchState(TypedDict):
    """
    State container tracking the research topic, accumulating findings list,
    generated exploratory questions, depth limit, and synthesized summary.
    """
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def iterative_research():
    """
    Constructs and executes the autonomous deep research loop.
    """

    # Node 1: Fact Finder / Researcher
    def research(state: ResearchState) -> dict:
        print(f"-> [Node: research] Deep Dive (Depth Level {state['iteration'] + 1}/{state['max_depth']})...")
        if state["iteration"] == 0:
            query = f"Give me 3 key technical facts about: {state['topic']}"
        else:
            query = (
                f"Based on these previous findings:\n{state['findings'][-1]}\n\n"
                f"Investigate this specific question in greater detail:\n{state['questions'][-1]}"
            )

        response = model.invoke(query)
        finding_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {
            "findings": [finding_text],
        }

    # Node 2: Question Generator
    def generate_questions(state: ResearchState) -> dict:
        print("-> [Node: generate_questions] Formulating follow-up inquiry...")
        response = model.invoke(
            f"Based on this research finding:\n{state['findings'][-1]}\n\n"
            f"What is one deeper, non-trivial question to explore further? Reply with just the question."
        )
        question_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        print(f"   Generated Question: {question_text.strip()}\n")
        return {
            "questions": [question_text.strip()],
            "iteration": state["iteration"] + 1,
        }

    # Node 3: Synthesizer
    def synthesize(state: ResearchState) -> dict:
        print("-> [Node: synthesize] Synthesizing all findings into comprehensive briefing...")
        all_findings = "\n\n".join(
            [f"--- Finding {i+1} ---\n{f}" for i, f in enumerate(state["findings"])]
        )
        response = model.invoke(
            f"Synthesize these depth-first research findings into a structured summary for topic: '{state['topic']}':\n\n{all_findings}"
        )
        summary_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {"summary": summary_text}

    # Conditional Exit Function
    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            return "synthesize"
        return "research"

    # Assemble Graph
    workflow = StateGraph(ResearchState)
    workflow.add_node("research", research)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("synthesize", synthesize)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "generate_questions")
    workflow.add_conditional_edges(
        "generate_questions",
        should_continue,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

    app = workflow.compile()

    print("=== Running Autonomous Depth-First Research Agent ===")
    initial_state = {
        "topic": "quantum computing applications in drug discovery",
        "findings": [],
        "questions": [],
        "max_depth": 2,
        "summary": "",
        "iteration": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"Research Topic   : {result['topic']}")
    print(f"Total Iterations : {result['iteration']}")
    print(f"Findings Count   : {len(result['findings'])}")
    print(f"\nFinal Executive Summary:\n{result['summary']}")
    print("=" * 60)


if __name__ == "__main__":
    iterative_research()
