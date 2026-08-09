from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class RouterState(TypedDict):
    query: str
    query_type: str
    response: str


def router():
    def classify_query(state: RouterState) -> dict:
        response = model.invoke(
            f"Classigy this query as 'question', 'command' or 'statement'"
            f"Reply iwth just the work. \n\n{state['query']}"
        )
        print("Response", response)
        return {"query_type": response.content[0].get("text").lower().strip()}

    def handle_question(state: RouterState) -> dict:
        response = model.invoke(f"Answer this question: {state['query']}")
        return {"response": f"[Answer] {response.content}"}

    def handle_command(state: RouterState) -> dict:
        return {
            "response": f"[Executing] I'll get back to you on this {state['query']}"
        }

    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged] Thanks for sharing {state['query']}"}

    def route_by_type(
        state: RouterState,
    ) -> Literal["question", "command", "statement"]:
        qt = state["query_type"]
        if "question" in qt:
            return "question"
        if "command" in qt:
            return "command"
        return "statement"

    # Create graph
    workflow = StateGraph(RouterState)

    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("handle_question", handle_question)
    workflow.add_node("handle_command", handle_command)
    workflow.add_node("handle_statement", handle_statement)

    # Add edges
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "question": "handle_question",
            "command": "handle_command",
            "statement": "handle_statement",
        },
    )
    workflow.add_edge("handle_question", END)
    workflow.add_edge("handle_command", END)
    workflow.add_edge("handle_statement", END)

    graph = workflow.compile()

    queries = [
        "Send an Email to John",
        "I love programming",
        "Who is the president of India." "Who is the PM of India.",
    ]

    for query in queries:
        response = graph.invoke({"query": query})
        print(f"Query : {query}")
        print(f"Type : {response['query_type']}")
        print(f"Response : {response['response']}\n\n")
        print("-" * 50 + "\n")


router()
