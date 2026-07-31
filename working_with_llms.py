from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 

load_dotenv()

def demo_init_chat_model_google():
   chat_model = init_chat_model(
        model="gemini-3.5-flash-lite",
        model_provider="google_genai"

   )
   return chat_model


def demo_message():
    chat_model = demo_init_chat_model_google()

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="what is the capital of France in one word time?")
    ]
    response = chat_model.invoke(messages)
    print(response.content[0].get("text"))

if __name__ == "__main__":
    demo_message()