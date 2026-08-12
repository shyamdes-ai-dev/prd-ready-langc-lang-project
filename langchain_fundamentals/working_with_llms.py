"""
Working with Chat Models & Message Roles
========================================
Demonstrates how to initialize chat models using the universal `init_chat_model`
factory method and structure multi-turn dialogues with explicit role-based messages
(`SystemMessage`, `HumanMessage`, `AIMessage`).
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Load API credentials from .env
load_dotenv()


def demo_init_chat_model_google():
    """
    Initializes a chat model using the unified `init_chat_model` abstraction.
    This pattern decouples the code from specific provider classes, making it
    easy to switch between Google, OpenAI, Anthropic, etc. by changing parameters.

    Returns:
        BaseChatModel: Configured chat model instance.
    """
    chat_model = init_chat_model(
        model="gemini-3.5-flash-lite",
        model_provider="google_genai",
    )
    return chat_model


def demo_message():
    """
    Demonstrates sending a sequence of structured role messages:
    - SystemMessage: Sets the persona, behavior, and constraints for the assistant.
    - HumanMessage : Represents the user query or instruction.
    """
    chat_model = demo_init_chat_model_google()

    # Define message payload with system context and human query
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="what is the capital of France in one word time?"),
    ]

    # Invoke chat model with the message list
    response = chat_model.invoke(messages)

    # Extract the generated text content from the response
    print("Response:", response.content[0].get("text") if isinstance(response.content, list) else response.content)


if __name__ == "__main__":
    demo_message()
