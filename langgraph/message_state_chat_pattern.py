from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import operator
from dotenv import load_dotenv


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def add_message(state: MessageState, message: BaseMessage) -> dict:
    return {"messages": message}

def call_model(state: MessageState) -> dict:
    llm = init_chat_model()
    response = llm.invoke(state["messages"])
    return {"messages": response}