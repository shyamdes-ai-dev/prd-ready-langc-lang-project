"""
Message-Centric Conversational State Pattern
============================================
Demonstrates the standard conversational state pattern in LangGraph using a list
of BaseMessage objects with an accumulating reducer (`operator.add`).

Pattern Benefits:
- Adheres to standard chat API protocols (OpenAI / Gemini / Anthropic message formats).
- Keeps the entire conversation history intact as new Human and AI messages are appended.
"""

import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# Load API credentials
load_dotenv()


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class MessageState(TypedDict):
    """
    Standard chat state holding the conversation history list.
    The reducer operator.add ensures incoming messages are appended to the list.
    """
    messages: Annotated[list[BaseMessage], operator.add]


def message_state():
    """
    Constructs and executes a message-driven chat graph.
    """
    llm = init_chat_model(
        model="gemini-3.5-flash", model_provider="google_genai"
    )

    # Chat node: invokes LLM with existing message history and returns the new response
    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Assemble graph
    graph = StateGraph(MessageState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    app = graph.compile()

    print("=== Invoking Message-Centric Chat Graph ===")
    initial_messages = [HumanMessage(content="Say Hello in Tagalog")]
    result = app.invoke({"messages": initial_messages})

    # Display dialogue exchange
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"Human: {msg.content}")
        else:
            ai_text = msg.content[0].get("text") if isinstance(msg.content, list) else msg.content
            print(f"AI   : {ai_text}")


if __name__ == "__main__":
    message_state()
