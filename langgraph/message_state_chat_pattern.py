from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import operator
from dotenv import load_dotenv

load_dotenv()


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def message_state():
    llm = model = init_chat_model(
        model="gemini-3.5-flash", model_provider="google_genai"
    )

    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessageState)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    app = graph.compile()
    result = app.invoke({"messages": [HumanMessage(content="Say Heloo in Tagalog")]})
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(
            f"{role}: {msg.content}"
            if role == "Human"
            else f"{role}: {msg.content[0].get('text')}"
        )


message_state()
