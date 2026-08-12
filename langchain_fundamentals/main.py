"""
Environment & Version Verification
==================================
Demonstrates how to inspect installed package versions for `langchain_core` and
`langgraph`, load environment variables securely via `python-dotenv`, and perform
a basic sanity check invocation using `ChatGoogleGenerativeAI`.
"""

import importlib.metadata
from dotenv import load_dotenv

# Retrieve version of langchain_core directly from the package
from langchain_core import __version__ as version_core
from langchain_google_genai import ChatGoogleGenerativeAI

# Retrieve the installed version of langgraph using importlib
version_graph = importlib.metadata.version("langgraph")

print(f"Langchain Core Version : {version_core}")
print(f"Langgraph Version      : {version_graph}")

# Load environment variables (e.g. GEMINI_API_KEY) from .env file
load_dotenv()


def invoke_llm():
    """
    Initializes a ChatGoogleGenerativeAI model instance and executes a single test prompt
    to verify that the API connection and credentials are functioning properly.
    """
    # Initialize the Google Gemini Chat Model with temperature=0 for deterministic output
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

    # Perform a simple string invocation
    response = llm.invoke("Say 'setup complete' in one word ")

    print("Raw response from Google Gemini:", response.content)
    print("Extracted text:", response.content[0]["text"])


if __name__ == "__main__":
    invoke_llm()
