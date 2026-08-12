"""
Sentiment-Adaptive Conversational Graph
=======================================
Demonstrates a multi-stage conversational agent that dynamically tunes its personality:
1. `analyze_sentiment`: Evaluates the emotional polarity of the user's message (positive, negative, neutral).
2. `generate_response`: Dynamically customizes its system prompt to match the detected sentiment
   (e.g., empathetic for negative, enthusiastic for positive, informative for neutral).
"""

import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# Load API credentials
load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class ConversationState(TypedDict):
    """
    Tracks accumulating conversation logs, detected sentiment polarity, and response count.
    """
    messages: Annotated[list, operator.add]
    sentiment: str
    response_count: int


def create_conversation_graph():
    """
    Builds and compiles the sentiment-adaptive conversation graph.
    """
    workflow = StateGraph(ConversationState)

    # Node 1: Sentiment Analysis
    def analyze_sentiment(state: ConversationState) -> dict:
        last_message = state["messages"][-1]
        response = model.invoke(
            [
                SystemMessage(
                    content="Classify sentiment strictly as: positive, negative, or neutral. "
                            "Reply with only that single lowercase word."
                ),
                HumanMessage(content=last_message),
            ]
        )
        sentiment_text = response.content[0].get("text") if isinstance(response.content, list) else response.content
        cleaned_sentiment = sentiment_text.lower().strip()
        print(f"[Sentiment Analyzer] Detected: '{cleaned_sentiment}'")
        return {"sentiment": cleaned_sentiment}

    # Node 2: Adaptive Response Generation
    def generate_response(state: ConversationState) -> dict:
        last_message = state["messages"][-1]
        sentiment = state["sentiment"]

        # Dynamic persona adjustment based on user emotional state
        system_prompts = {
            "positive": "Respond enthusiastically and celebrate the user's positive energy!",
            "negative": "Respond empathetically, validate their frustration, and offer supportive guidance.",
            "neutral": "Respond informatively, helpfully, and clearly.",
        }
        active_prompt = system_prompts.get(sentiment, system_prompts["neutral"])

        response = model.invoke(
            [
                SystemMessage(content=active_prompt),
                HumanMessage(content=last_message),
            ]
        )
        ai_reply = response.content[0].get("text") if isinstance(response.content, list) else response.content
        return {
            "messages": [f"AI: {ai_reply}"],
            "response_count": 1,
        }

    # Register Nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("generate_response", generate_response)

    # Wire Edges: START -> analyze_sentiment -> generate_response -> END
    workflow.add_edge(START, "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


def conversation_graph():
    """
    Tests the sentiment-adaptive graph with positive, negative, and neutral scenarios.
    """
    graph = create_conversation_graph()

    test_messages = [
        "I just got promoted at work! I am so thrilled! :)",
        "My computer crashed and I lost all my unsaved work before the deadline.",
        "What is the average surface temperature of Mars?",
    ]

    print("\n=== Running Sentiment-Adaptive Conversational Agent ===")
    for msg in test_messages:
        print("\n" + "=" * 60)
        print(f"User Message: {msg}")
        result = graph.invoke(
            {
                "messages": [f"Human: {msg}"],
                "sentiment": "",
                "response_count": 0,
            }
        )
        print(f"Detected Sentiment: {result['sentiment']}")
        print(f"Response:\n{result['messages'][-1]}")


if __name__ == "__main__":
    conversation_graph()
