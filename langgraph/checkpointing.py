"""
State Checkpointing & Multi-Turn Persistence
============================================
Demonstrates state persistence and thread isolation in LangGraph:
1. `MemorySaver` : Ephemeral in-memory checkpointer suitable for development and testing.
2. `SqliteSaver` : Durable disk-based database checkpointer suitable for production session storage.
3. Thread Isolation (`thread_id`): Isolating state across multi-tenant sessions and retrieving
   historic state via `app.get_state(config)`.
"""

import operator
import tempfile
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# Load API credentials
load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------
class ChatState(TypedDict):
    """
    Standard state tracking conversational turns with an accumulating message reducer.
    """
    messages: Annotated[list[BaseMessage], operator.add]


def memory_saver():
    """
    Demonstrates in-memory checkpointing using `MemorySaver`.
    Maintains multi-turn context within the application process across discrete invocations
    sharing the same `thread_id`.
    """
    print("\n=== 1. In-Memory Checkpointing (MemorySaver) ===")

    def chat(state: ChatState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    # Compile graph with in-memory checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Thread configuration to distinguish this user session
    config = {"configurable": {"thread_id": "thread_user_101"}}

    # Multi-Turn Interaction: Turn 1
    print("[Turn 1] User: 'My name is Shyam'")
    result1 = app.invoke(
        {"messages": [HumanMessage(content="My name is Shyam")]},
        config=config,
    )
    ai_text1 = result1["messages"][-1].content[0].get("text") if isinstance(result1["messages"][-1].content, list) else result1["messages"][-1].content
    print(f"         AI  : {ai_text1}\n")

    # Multi-Turn Interaction: Turn 2 (Model remembers name from Checkpointed Thread)
    print("[Turn 2] User: 'What is my name?'")
    result2 = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config,
    )
    ai_text2 = result2["messages"][-1].content[0].get("text") if isinstance(result2["messages"][-1].content, list) else result2["messages"][-1].content
    print(f"         AI  : {ai_text2}\n")

    # Inspect total checkpointed state history
    state_snapshot = app.get_state(config=config)
    print(f"Checkpointer History Count: {len(state_snapshot.values['messages'])} message(s) stored.")


def sqlite_persistence():
    """
    Demonstrates durable disk persistence using `SqliteSaver`.
    Allows the application to preserve conversation states across server restarts
    or independent process runs by persisting checkpoints to an SQLite database file.
    """
    print("\n=== 2. Durable SQLite Checkpointing (SqliteSaver) ===")

    def chat(state: ChatState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    # Create temporary database file for demonstration
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Persisting session checkpoints to SQLite DB: {db_path}")

    # Process Session 1: Store secret credentials in thread
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent_session_42"}}

        app.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember this confidential token: ALPHA-9988")
                ]
            },
            config=config,
        )
        print("[Session 1] Stored secret token ALPHA-9988 into SQLite thread.")

    # Process Session 2: Re-open SQLite database in a fresh session and query the secret
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent_session_42"}}

        print("[Session 2] Reconnected to SQLite. Asking: 'What was the secret token?'")
        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret token?")]},
            config=config,
        )
        ai_text = result["messages"][-1].content[0].get("text") if isinstance(result["messages"][-1].content, list) else result["messages"][-1].content
        print(f"AI Response : {ai_text}")


def main():
    """Run both memory and SQLite checkpointing demos."""
    memory_saver()
    sqlite_persistence()


if __name__ == "__main__":
    main()
