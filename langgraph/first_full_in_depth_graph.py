from langchain_core.messages import content
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import operator

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")

class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    sentiment: str
    response_count: int


def create_conversation_graph():
    workflow = StateGraph(ConversationState)

    def analyze_sentiment(state: ConversationState) -> dict:
        last_message = state['messages'][-1]

        response = model.invoke([
            SystemMessage(content="classify sentiment as : postive, negative or neutral. Reply just with only single word"),
            HumanMessage(content=last_message)
        ])
        print("=" * 60)
        print("sentiment analyser output : ", response)
        
        return {"sentiment": response.content[0].get("text")}

    def generate_response(state: ConversationState) -> dict:
        last_message = state['messages'][-1]
        sentiment = state['sentiment']
        
        system_prompt = {
            "positive": "Respond enthusiastically and build on their positive energy",
            "negative": "Respond empathetically and offer support",
            "neutral": "Respond helpfully and informatively"
        }
        prompt = system_prompt.get(sentiment, system_prompt["neutral"])
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_message)
        ])
        return {"messages": [f"AI: {response.content[0].get('text')}"],"response_count":1}

    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
                   

def conversation_graph():
    graph = create_conversation_graph()
    test_messages = [
                     "I just got promoted at Work! I am so happy :)", 
                     "My computer  Crashed and I lost all my work",
                     "What's the weatehr like today?"
                    ]
    for message in test_messages:
        response = graph.invoke({
            "messages": [f"Human: {message}"],
            "sentiment": "",
            "response_count": 0
        })
        print("*"*30)
        print(response["messages"])
        print("=" * 60)

if __name__ == "__main__":
    conversation_graph()
        
