"""
Basic LLM Invocation
====================
Demonstrates the most straightforward, minimal pattern for initializing and invoking
a chat model using `ChatGoogleGenerativeAI` and `python-dotenv`.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 1: Load environment variables (such as GEMINI_API_KEY) from .env
load_dotenv()


def main():
    """
    Initializes a Gemini Flash model and executes a simple question prompt.
    """
    # Step 2: Instantiate the chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.7,
    )

    # Step 3: Invoke the model with a plain text prompt
    prompt = "Explain in one sentence why LangChain is useful for AI developers."
    response = llm.invoke(prompt)

    print("Prompt:", prompt)
    print("Response:", response.content[0].get("text") if isinstance(response.content, list) else response.content)


if __name__ == "__main__":
    main()
