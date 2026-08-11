from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator
import tempfile
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def memory_saver():
    """In-memory checkpointing for development"""

    def chat(state: ChatState) -> dict: 
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat)

    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("Memory Saver (Multi-turn conversation): \n")

    # Need a config object to store the checkpoint id 
    config = {"configurable": {"thread_id": "thread_1"}}
    
    # Turn 1
    result = app.invoke(
        {"messages": [HumanMessage(content="My Name is Shyam?")]},
        config=config
    )

    print(f"Turn 1 - AI: {result['messages'][-1].content[0].get('text')}\n")

    # Turn 2
    result = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config
    )

    print(f"Turn 2 - AI: {result['messages'][-1].content[0].get('text')}\n")

    # Check full history
    state = app.get_state(config=config)
    print("Full conversation history: ", len(state.values["messages"]))


def sqlite_persistence():
    """ SQLite persistence for durable storage."""


    def chat(state: ChatState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages":[response]}
    
    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat)

    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    
    # Create Temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"\n SQLITE Persistence:")
    print(f"Database: {db_path}")

    # Create SqliteSaver
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {
                "messages": [HumanMessage(content="Remeber: The secreate code is ALPHA-123")]
            },
            config=config
        )
        print(f"Session 1 - Stored secret Code")

         

        