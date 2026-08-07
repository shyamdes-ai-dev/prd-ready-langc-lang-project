from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import operator
from dotenv import load_dotenv

load_dotenv()


class AccumulatingState(TypedDict):
    """
    Accumulating state.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    count: Annotated[int, operator.add]


def accumulating_state():
    def step_one(state: AccumulatingState) -> dict:
        return {"messages": [HumanMessage(content="Hello")], "count": 1}

    def step_two(state: AccumulatingState) -> dict:
        return {"messages": [AIMessage(content="Hi")], "count": 1}

    graph = StateGraph(AccumulatingState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app = graph.compile()

    # result = app.invoke({"messages": [HumanMessage(content="Starting message")], "count": 0})
    # print(result["messages"])
    # print(result["count"])

    for chunk in app.stream(
        {"messages": [HumanMessage(content="Starting message")], "count": 0},
        stream_mode="values",
    ):
        print(chunk)
