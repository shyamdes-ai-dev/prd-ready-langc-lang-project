from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import operator
from dotenv import load_dotenv

load_dotenv()


#Basic state

class SimpleState(TypedDict):
    input: str
    output: str
    step: str    

def simple_graph():
    
    #define node functions
    def process(state: SimpleState) -> dict:
        return {"output": state["input"].upper(), "step": state["step"] + 1}
    
    graph = StateGraph(SimpleState)

    graph.add_node("process", process)

    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    
    #Excecute graph/ Compile
    workflow = graph.compile()
    
    png_bytes = workflow.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved to graph.png")

    result = workflow.invoke({"input": "hello world", "step": 0})
    print(f" Input: {result['input']}, Output: {result['output']}, Step: {result['step']}")

simple_graph()